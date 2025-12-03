import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------
# Helpers / Preproc config
# -----------------------
TARGET_MAP = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "PTSD"
}

# categorical options (ambil dari untitled1.py asumsi umum;
# kamu bisa tambahkan/match dengan nilai asli dataset jika perlu)
COUNTRY_OPTIONS = ["USA", "Other"]
GENDER_OPTIONS = ["Female", "Male", "Other"]
EXERCISE_OPTIONS = ["Low", "Moderate", "High"]
DIET_OPTIONS = ["Junk Food", "Balanced", "Vegetarian", "Other"]
STRESS_OPTIONS = ["Low", "Moderate", "High"]

NUM_COLS = [
    "Age",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score"
]

CATEGORICAL_COLS = [
    "Country",
    "Gender",
    "Exercise Level",
    "Diet Type",
    "Stress Level"
]

# -----------------------
# Load Model & Scaler
# -----------------------
@st.cache_resource
def load_model(path="gb_model.pkl"):
    if not os.path.exists(path):
        return None, f"Model file not found at {path}"
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, None

@st.cache_resource
def load_scaler(path="scaler.pkl"):
    if not os.path.exists(path):
        return None, "Scaler file not found (optional)."
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler, None

model, model_err = load_model()
scaler, scaler_err = load_scaler()

st.title("Mental Health Condition Predictor")
st.write("Aplikasi prediksi berdasarkan model Gradient Boosting yang dilatih. Pastikan `gb_model.pkl` ada di folder yang sama.")

if model_err:
    st.error(model_err)
else:
    st.success("Model berhasil dimuat.")

    if scaler is None:
        st.warning("Scaler tidak ditemukan. Karena model tree-based (GradientBoosting) tidak selalu memerlukan scaling, aplikasi akan tetap berjalan — namun hasil bisa berbeda dari lingkungan training jika scaler sebenarnya digunakan saat training. Kalau kamu punya file `scaler.pkl`, letakkan di folder yang sama untuk hasil yang lebih konsisten.")

    st.markdown("---")
    st.header("Masukkan data pasien (satu sampel)")

    # --- Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=120, value=30)
        sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        work_hours = st.number_input("Work Hours per Week", min_value=0.0, max_value=168.0, value=40.0, step=1.0)
    with col2:
        screen_time = st.number_input("Screen Time per Day (Hours)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        social_interaction = st.slider("Social Interaction Score (0-10)", 0.0, 10.0, 5.0, 0.5)
        happiness = st.slider("Happiness Score (0-10)", 0.0, 10.0, 5.0, 0.5)

    country = st.selectbox("Country", options=COUNTRY_OPTIONS)
    gender = st.selectbox("Gender", options=GENDER_OPTIONS)
    exercise = st.selectbox("Exercise Level", options=EXERCISE_OPTIONS)
    diet = st.selectbox("Diet Type", options=DIET_OPTIONS)
    stress = st.selectbox("Stress Level", options=STRESS_OPTIONS)

    if st.button("Predict"):
        # build one-row dataframe
        row = {
            "Age": age,
            "Sleep Hours": sleep,
            "Work Hours per Week": work_hours,
            "Screen Time per Day (Hours)": screen_time,
            "Social Interaction Score": social_interaction,
            "Happiness Score": happiness,
            "Country": country,
            "Gender": gender,
            "Exercise Level": exercise,
            "Diet Type": diet,
            "Stress Level": stress
        }
        X = pd.DataFrame([row])

        # --- Preprocessing (mengikuti untitled1.py) ---
        # 1) Feature engineering
        # Clip Sleep Hours
        X["Sleep Hours"] = X["Sleep Hours"].clip(lower=2, upper=10)

        X["lifestyle_risk_score"] = (
            (10 - X["Sleep Hours"]) +
            (X["Work Hours per Week"] / 10.0) +
            (X["Screen Time per Day (Hours)"]) +
            (10 - X["Social Interaction Score"])
        )

        X["work_life_ratio"] = X["Work Hours per Week"] / X["Sleep Hours"].replace(0, 1e-6)
        X["social_wellbeing"] = (X["Social Interaction Score"] + X["Happiness Score"]) / 2.0

        # 2) One-hot encoding categories (drop_first=True saat training)
        # We will mimic get_dummies + drop_first by creating dummies and dropping the first category per feature (alphabetic first isn't necessarily the original drop_first; this is best-effort).
        # Safer approach: produce dummies, then reindex to model expected features.
        X_dummies = pd.get_dummies(X[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, drop_first=True)
        X_num = X[NUM_COLS + ["lifestyle_risk_score", "work_life_ratio", "social_wellbeing"]]
        X_proc = pd.concat([X_num, X_dummies], axis=1)

        # 3) Align features with model.feature_names_in_ if available
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        else:
            # fallback: try to infer from model n_features_in_
            if hasattr(model, "n_features_in_"):
                n = model.n_features_in_
                # if counts match, assume numeric features order + dummies as created
                # Otherwise warn user
                if X_proc.shape[1] == n:
                    expected_cols = list(X_proc.columns)
                else:
                    expected_cols = list(X_proc.columns)  # best-effort
                    st.warning("Model does not expose feature names (feature_names_in_). We will attempt best-effort alignment.")
            else:
                expected_cols = list(X_proc.columns)
                st.warning("Model metadata missing. Using current feature set as-is (best-effort).")

        # Ensure all expected columns exist in X_proc, if not add missing with 0
        for c in expected_cols:
            if c not in X_proc.columns:
                X_proc[c] = 0

        # Reorder to expected_cols
        X_proc = X_proc[expected_cols]

        # 4) Scaling if scaler available
        if scaler is not None:
            try:
                # scaler expects numeric columns in training order. We attempt to scale only numeric columns that exist in scaler.mean_.
                # If scaler was fit on a subset, create matching array.
                numeric_cols_for_scaler = None
                if hasattr(scaler, "mean_"):
                    # Try to detect columns used by scaler:
                    # Many people save full StandardScaler — but without column names. We'll assume scaler was fit on NUM_COLS + engineered features.
                    numeric_cols_for_scaler = [
                        "Age", "Sleep Hours", "Work Hours per Week",
                        "Screen Time per Day (Hours)", "Social Interaction Score",
                        "Happiness Score", "lifestyle_risk_score",
                        "work_life_ratio", "social_wellbeing"
                    ]
                    # If all those exist in X_proc, scale them in-place
                    miss = [c for c in numeric_cols_for_scaler if c not in X_proc.columns]
                    if len(miss) == 0:
                        X_proc[numeric_cols_for_scaler] = scaler.transform(X_proc[numeric_cols_for_scaler])
                    else:
                        st.warning(f"Scaler was found but expected numeric columns missing: {miss}. Skipping scaler transform.")
                else:
                    st.warning("Scaler loaded but does not have mean_/scale_ attributes. Skipping.")
            except Exception as e:
                st.warning(f"Terjadi error saat menerapkan scaler: {e}. Melanjutkan tanpa scaling.")
        else:
            st.info("Tidak ada scaler, melanjutkan tanpa scaling numeric features.")

        # 5) Prediction
        try:
            pred = model.predict(X_proc)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_proc)[0]
            label = TARGET_MAP.get(int(pred[0]), str(pred[0]))

            st.subheader("Hasil Prediksi")
            st.write(f"Prediksi label: **{label}**")
            if proba is not None:
                # present as dataframe
                proba_df = pd.DataFrame([proba], columns=[TARGET_MAP[i] for i in range(len(proba))])
                st.write("Probabilitas tiap kelas:")
                st.dataframe(proba_df.T.rename(columns={0:"Probability"}))
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

