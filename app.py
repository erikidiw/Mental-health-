import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# 🚀 Load Artifacts (Pipeline, Encoders, etc.)
# ==========================
try:
    # Memuat semua objek yang diperlukan dari file .pkl
    artifacts = joblib.load('pipeline_artifacts.pkl')
    pipeline = artifacts['pipeline']
    label_encoders = artifacts['label_encoders']
    target_encoder = artifacts['target_encoder']
    ordinal_mapper = artifacts['ordinal_mapper']
    feature_cols = artifacts['feature_cols']
    
    st.sidebar.success("✅ Model dan Preprocessor berhasil dimuat.")
    
except FileNotFoundError:
    st.sidebar.error("❌ File 'pipeline_artifacts.pkl' tidak ditemukan. Jalankan skrip 'create_pipeline.py' terlebih dahulu.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat artifacts: {e}")
    st.stop()


# ==========================
# 🔧 PREPROCESSING & PREDICTION FUNCTION
# ==========================

def preprocess_and_predict(input_data):
    """
    Mengubah input mentah menjadi data yang siap diprediksi 
    menggunakan encoder yang sudah dilatih, lalu melakukan prediksi.
    """
    # 1. Konversi dictionary ke DataFrame
    df_single = pd.DataFrame([input_data])
    
    # 2. Pastikan urutan dan nama kolom sesuai dengan saat training
    df_single = df_single[feature_cols]

    # 3. Cleaning (meniru langkah awal training)
    for col in ['Sleep Duration', 'Financial Stress']:
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str).str.replace("'", "").str.strip()
            
    df_single['Financial Stress'] = df_single['Financial Stress'].replace('?', '0')

    # 4. Ordinal Mapping (menggunakan mapper yang sudah dilatih)
    df_single[ordinal_mapper.cols] = ordinal_mapper.transform(df_single)
    
    # 5. Label Encoding (menggunakan encoder yang sudah dilatih)
    for col, le in label_encoders.items():
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str)
            # Menangani kategori yang tidak pernah dilihat (unseen categories)
            df_single[col] = df_single[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # 6. Target Encoding (menggunakan encoder yang sudah dilatih)
    target_cols = ['City', 'Profession']
    df_single[target_cols] = target_encoder.transform(df_single[target_cols])
    
    # 7. Konversi semua kolom yang tersisa menjadi float
    df_processed = df_single.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # 8. Prediksi melalui Pipeline (akan melakukan Scaling + Classification)
    prediction = pipeline.predict(df_processed)[0]
    return prediction


# ==========================
# 🧠 STREAMLIT UI
# ==========================

st.title("🩺 Mental Health Depression Prediction App")
st.write("Masukkan data Anda lalu klik prediksi untuk melihat hasilnya.")

# Karena kita tidak memiliki akses ke data asli di sini, kita tentukan opsi secara manual 
# sesuai dengan file dataset2.py Anda
df_placeholder = pd.DataFrame(columns=feature_cols)

st.sidebar.header("Input Data Mahasiswa")

# Input Form
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Others'])
city = st.sidebar.text_input("City (Contoh: Mumbai)")
profession = st.sidebar.text_input("Profession (Contoh: Student)")
age = st.sidebar.number_input("Age", min_value=10, max_value=80, value=25, step=1)
cgpa = st.sidebar.number_input("CGPA", min_value=2.0, max_value=10.0, value=7.5, step=0.1)
hours = st.sidebar.number_input("Work/Study Hours", min_value=0, max_value=20, value=5, step=1)
sleep = st.sidebar.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others'])
diet = st.sidebar.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy'])
degree = st.sidebar.selectbox("Degree", ['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'Others'])
social = st.sidebar.selectbox("Social Weakness", ['No', 'Yes'])
suicide = st.sidebar.selectbox("Have suicidal thoughts?", ["No", "Yes"])
financial = st.sidebar.selectbox("Financial Stress (1-5)", ["1.0", "2.0", "3.0", "4.0", "5.0", '?'])
history = st.sidebar.selectbox("Family History of Mental Illness", ["No", "Yes"])
academic = st.sidebar.number_input("Academic Pressure (0-5)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
satisfaction = st.sidebar.number_input("Study Satisfaction (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)


if st.sidebar.button("🔍 Predict"):
    
    # Kumpulkan input data sesuai nama kolom training
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
        "Study Satisfaction": satisfaction,
    }

    # Prediksi
    prediction = preprocess_and_predict(input_data)

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Risiko Tinggi (Depresi). Sebaiknya segera cari bantuan profesional.")
        st.markdown("Tingkat Risiko: **DEPRESI** (1)")
    else:
        st.success("💚 Risiko Rendah (Normal). Pertahankan pola hidup seimbang.")
        st.markdown("Tingkat Risiko: **TIDAK DEPRESI** (0)")
