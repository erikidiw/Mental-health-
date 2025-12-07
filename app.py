import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# 🔧 Custom Preprocessing Classes (Wajib ada untuk memuat PKL)
# ==========================
class CustomOrdinalMapper:
    def __init__(self, mappings):
        self.mappings = {col: map_dict for col, map_dict in mappings}
        self.cols = [col for col, _ in mappings]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Fungsi ini akan digantikan oleh transform asli dari objek yang dimuat
        pass

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
    
    st.success("✅ Model dan Preprocessor berhasil dimuat.")
    
except FileNotFoundError:
    st.error("❌ File 'pipeline_artifacts.pkl' tidak ditemukan. Jalankan skrip 'create_pipeline.py' terlebih dahulu.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal memuat artifacts: {e}")
    st.stop()


# ==========================
# 🔧 PREPROCESSING & PREDICTION FUNCTION
# ==========================

def preprocess_and_predict(input_data):
    """
    Mengubah input mentah menjadi data yang siap diprediksi 
    menggunakan encoder yang sudah dilatih, lalu melakukan prediksi.
    """
    df_single = pd.DataFrame([input_data])
    df_single = df_single[feature_cols]

    # Cleaning
    for col in ['Sleep Duration', 'Financial Stress']:
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str).str.replace("'", "").str.strip()
            
    df_single['Financial Stress'] = df_single['Financial Stress'].replace('?', '0')

    # Ordinal Mapping
    df_single[ordinal_mapper.cols] = ordinal_mapper.transform(df_single)
    
    # Label Encoding
    for col, le in label_encoders.items():
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str)
            df_single[col] = df_single[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Target Encoding
    target_cols = ['City', 'Profession']
    df_single[target_cols] = target_encoder.transform(df_single[target_cols])
    
    # Konversi ke float
    df_processed = df_single.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # Prediksi melalui Pipeline
    prediction = pipeline.predict(df_processed)[0]
    return prediction


# ==========================
# 🧠 STREAMLIT UI
# ==========================

st.title("🩺 Sistem Prediksi Risiko Depresi Mahasiswa")
st.write("Masukkan data kamu. Lalu tekan tombol prediksi di bawah.")

# Menggunakan kolom untuk tata letak yang lebih baik
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Informasi Dasar")
    gender = st.selectbox("Jenis Kelamin", ['Male', 'Female', 'Others'])
    age = st.number_input("Umur", min_value=10, max_value=80, value=25, step=1)
    degree = st.selectbox("Jenjang Pendidikan (Degree)", ['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'Others'])
    city = st.text_input("Kota Tinggal (Contoh: Mumbai)")
    profession = st.text_input("Pekerjaan (Contoh: Student)")
    
with col2:
    st.subheader("Faktor Akademik & Kehidupan")
    cgpa = st.number_input("Rata-rata IPK (CGPA)", min_value=2.0, max_value=10.0, value=7.5, step=0.1)
    hours = st.number_input("Jam Belajar/Kerja per hari", min_value=0, max_value=20, value=5, step=1)
    sleep = st.selectbox("Durasi Tidur", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others'])
    diet = st.selectbox("Kebiasaan Makan (Dietary Habits)", ['Healthy', 'Moderate', 'Unhealthy'])
    
with col3:
    st.subheader("Faktor Risiko Mental")
    academic = st.slider("Tekanan Akademik (1=Rendah, 5=Tinggi)", min_value=1, max_value=5, value=3, step=1)
    satisfaction = st.slider("Kepuasan Belajar (1=Rendah, 5=Tinggi)", min_value=1, max_value=5, value=4, step=1)
    
    # --- PERUBAHAN STRES KEUANGAN DI SINI ---
    financial = st.slider(
        "Stres Keuangan (1=Rendah, 5=Tinggi)", 
        min_value=1, 
        max_value=5, 
        value=3, 
        step=1 # Hanya bilangan bulat
    )
    # --- AKHIR PERUBAHAN ---
    
    social = st.selectbox("Kelemahan Sosial (Social Weakness)", ['No', 'Yes'])
    history = st.selectbox("Riwayat Mental Keluarga", ["No", "Yes"])
    suicide = st.selectbox("Pernah terpikir Bunuh Diri?", ["No", "Yes"])


# Tombol Prediksi di luar kolom agar posisinya tunggal
st.markdown("---")
if st.button("🔍 Prediksi Tingkat Risiko"):
    
    # Kumpulkan input data
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
        "Financial Stress": str(financial) + ".0", # Format integer slider menjadi string "X.0" agar sesuai dengan mapper training
        "Family History of Mental Illness": history,
        "Academic Pressure": academic,
        "Study Satisfaction": satisfaction,
    }

    # Prediksi
    prediction = preprocess_and_predict(input_data)

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Risiko Tinggi (Depresi). Segera cari bantuan profesional.")
        st.write("Tingkat Risiko: DEPRESI (1)")
    else:
        st.success("💚 Risiko Rendah (Normal). Pertahankan pola hidup seimbang.")
        st.write("Tingkat Risiko: TIDAK DEPRESI (0)")
