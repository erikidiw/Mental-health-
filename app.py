import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier


# ==========================
# 🔧 Load Dataset for Encoding Reference
# ==========================
df = pd.read_csv("student_depression_dataset.csv")   # harus ada di folder yang sama


# ==========================
# 🔧 PREPROCESSING FUNCTION
# ==========================

# Global encoders (will be fitted during model training)
label_encoders = {}
target_encoder = None
scaler = StandardScaler()

def preprocess_dataframe(df_input, fit_encoders=False):
    """Preprocess entire dataframe for training"""
    df_copy = df_input.copy()
    
    # -------- Drop Columns --------
    drop_cols = ['Work Pressure', 'Job Satisfaction']
    if 'id' in df_copy.columns:
        drop_cols.append('id')
    if 'Depression' in df_copy.columns:
        target = df_copy['Depression']
    else:
        target = None
    df_copy = df_copy.drop(columns=[col for col in drop_cols if col in df_copy.columns])
    
    # -------- Encoding Rules --------
    ordinal_mapping = {
        "Sleep Duration": {
            "Less than 5 hours": 1,
            "5-6 hours": 2,
            "7-8 hours": 3,
            "More than 8 hours": 4,
            "Others": 0
        },
        "Financial Stress": {
            "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4, "5.0": 5, "?": 0
        },
        "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
        "Family History of Mental Illness": {"No": 0, "Yes": 1}
    }
    
    for col, mapping in ordinal_mapping.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(mapping).fillna(0)
    
    # Label Encoding
    label_cols = ['Gender', 'Dietary Habits', 'Degree']
    
    for col in label_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str)
            if fit_encoders:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    df_copy[col] = label_encoders[col].transform(df_copy[col])
    
    # Target Encoding
    target_cols = ['City', 'Profession']
    if fit_encoders and target is not None:
        global target_encoder
        target_encoder = TargetEncoder()
        df_copy[target_cols] = target_encoder.fit_transform(df_copy[target_cols], target)
    else:
        if target_encoder is not None:
            df_copy[target_cols] = target_encoder.transform(df_copy[target_cols])
    
    # Ensure Academic Pressure and Study Satisfaction are numeric
    if "Academic Pressure" in df_copy.columns:
        df_copy["Academic Pressure"] = pd.to_numeric(df_copy["Academic Pressure"], errors='coerce').fillna(0)
    if "Study Satisfaction" in df_copy.columns:
        df_copy["Study Satisfaction"] = pd.to_numeric(df_copy["Study Satisfaction"], errors='coerce').fillna(0)
    
    # Scaling
    feature_cols = df_copy.columns
    if fit_encoders:
        df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])
    else:
        df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])
    
    return df_copy

def preprocess_input(data):
    """Preprocess single input dictionary for prediction"""
    # Convert dict to single-row dataframe
    df_single = pd.DataFrame([data])
    
    # Use the dataframe preprocessing function
    result = preprocess_dataframe(df_single, fit_encoders=False)
    
    return result



# ==========================
# 🚀 Load Model
# ==========================
# Prepare training data
train_df = df.drop(columns=["Depression"])

# Preprocess training data and fit encoders
X_train = preprocess_dataframe(train_df, fit_encoders=True)
y_train = df["Depression"]

model = RandomForestClassifier()
model.fit(X_train, y_train)


# ==========================
# 🧠 STREAMLIT UI
# ==========================

st.title("🩺 Mental Health Depression Prediction App")
st.write("Masukkan data kamu lalu klik predict untuk melihat hasilnya.")


# Input Form
gender = st.selectbox("Gender", df["Gender"].unique())
city = st.selectbox("City", df["City"].unique())
profession = st.selectbox("Profession", df["Profession"].unique())
age = st.number_input("Age", min_value=10, max_value=80, step=1)
cgpa = st.number_input("CGPA", min_value=2.0, max_value=10.0, step=0.1)
hours = st.number_input("Work/Study Hours", min_value=0, max_value=20, step=1)
sleep = st.selectbox("Sleep Duration", df["Sleep Duration"].unique())
financial = st.selectbox("Financial Stress (1-5)", ["1.0", "2.0", "3.0", "4.0", "5.0"])
diet = st.selectbox("Dietary Habits", df["Dietary Habits"].unique())
degree = st.selectbox("Degree", df["Degree"].unique())
social = st.selectbox("Social Weakness", df["Social Weakness"].unique())
suicide = st.selectbox("Have suicidal thoughts?", ["Yes", "No"])
history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
academic = st.number_input("Academic Pressure (0-5)", min_value=0.0, max_value=5.0, step=0.1)
satisfaction = st.number_input("Study Satisfaction (0-5)", min_value=0.0, max_value=5.0, step=0.1)


if st.button("🔍 Predict"):

    input_data = {
        "Gender": gender,
        "City": city,
        "Profession": profession,
        "Age": age,
        "CGPA": cgpa,
        "Work/Study Hours": hours,
        "Sleep Duration": sleep,
        "Dietary Habits": diet,
        "Degree": degree,
        "Social Weakness": social,
        "Have you ever had suicidal thoughts ?": suicide,
        "Financial Stress": financial,
        "Family History of Mental Illness": history,
        "Academic Pressure": academic,
        "Study Satisfaction": satisfaction
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]

    if prediction == 1:
        st.error("⚠️ Kamu menunjukkan indikasi depresi. Sebaiknya konsultasi dengan profesional.")
    else:
        st.success("💚 Kamu tidak menunjukkan tanda depresi. Tetap jaga kesehatan mental ya!")
