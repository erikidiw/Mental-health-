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
# Load Model
# -----------------------
@st.cache_resource
def load_model(model_path="gb_model.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

st.title("Mental Health Condition Predictor")
if model is None:
    st.error("Model `gb_model.pkl` tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()
else:
    st.success("Model berhasil dimuat ✔")

# -----------------------
# Load Scaler
# -----------------------
@st.cache_resource
def load_scaler(scaler_path="scaler.pkl"):
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None

scaler = load_scaler()

if scaler is None:
    st.warning("Scaler tidak ditemukan. Silakan upload `scaler.pkl` jika diperlukan:")

    scaler_file = st.file_uploader("🔼 Upload scaler.pkl", type=["pkl"])
    if scaler_file is not None:
        try:
            scaler = pickle.load(scaler_file)
            st.success("Scaler berhasil dimuat dari upload ✔")
        except:
            st.error("Gagal membaca scaler.pkl. Pastikan file benar!")

st.markdown("---")
st.header("Masukkan Data Pasien")

# -----------------------
# INPUT FORM
# -----------------------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=120, value=30)
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    work_hours = st.number_input("Work Hours per Week", min_value=0.0, max_value=168.0, value=40.0)

with col2:
    screen_time = st.number_input("Screen Time per Day (Hours)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
    social_interaction = st.slider("Social Interaction Score (0-10)", 0.0, 10.0, 5.0)
    happiness = st.slider("Happiness Score (0-10)", 0.0, 10.0, 5.0)

country = st.selectbox("Country", COUNTRY_OPTIONS)
gender = st.selectbox("Gender", GENDER_OPTIONS)
exercise = st.selectbox("Exercise Level", EXERCISE_OPTIONS)
diet = st.selectbox("Diet Type", DIET_OPTIONS)
stress = st.selectbox("Stress Level", STRESS_OPTIONS)

# -----------------------
# PREDICTION BUTTON
# -----------------------
if st.button("Predict"):
    X = pd.DataFrame([{
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
    }])

    # Preprocessing
    X["Sleep Hours"] = X["Sleep Hours"].clip(2, 10)
    X["lifestyle_risk_score"] = (10 - X["Sleep Hours"]) + (X["Work Hours per Week"] / 10) + \
                                (X["Screen Time per Day (Hours)"]) + (10 - X["Social Interaction Score"])
    X["work_life_ratio"] = X["Work Hours per Week"] / X["Sleep Hours"].replace(0, 1e-6)
    X["social_wellbeing"] = (X["Social Interaction Score"] + X["Happiness Score"]) / 2

    X_dummies = pd.get_dummies(X[CATEGORICAL_COLS], drop_first=True)
    X_num = X[NUM_COLS + ["lifestyle_risk_score", "work_life_ratio", "social_wellbeing"]]
    X_proc = pd.concat([X_num, X_dummies], axis=1)

    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in X_proc.columns:
                X_proc[col] = 0
        X_proc = X_proc[model.feature_names_in_]

    # Apply scaler if exists
    if scaler is not None:
        try:
            X_proc[X_proc.columns] = scaler.transform(X_proc[X_proc.columns])
        except:
            st.warning("Scaler tidak cocok. Prediksi tanpa scaling.")

    try:
        pred = model.predict(X_proc)[0]
        label = TARGET_MAP.get(pred, "Unknown")

        st.subheader("🎯 Hasil Prediksi")
        st.success(f"Prediksi Kondisi Mental: **{label}**")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
