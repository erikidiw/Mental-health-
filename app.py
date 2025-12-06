import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier # Import model yang digunakan

# --- 1. Load Model ---
# Model yang disimpan adalah Random Forest yang dilatih pada 3 fitur.
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("✅ Model 'model.pkl' berhasil dimuat.")
except FileNotFoundError:
    st.sidebar.error("❌ File 'model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    model = None
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    model = None


# --- 2. Fungsi Prediksi ---
def predict_risk(model, data):
    """
    Melakukan prediksi risiko menggunakan model yang dimuat.
    Data input harus memiliki 3 kolom: 'Academic Pressure', 'CGPA', 'Study Satisfaction'.
    """
    if model is None:
        return "Model Gagal Dimuat"

    # Pastikan urutan dan nama kolom sesuai dengan saat pelatihan model
    features = ['Academic Pressure', 'CGPA', 'Study Satisfaction']
    input_df = pd.DataFrame([data], columns=features)

    # Prediksi
    prediction = model.predict(input_df)[0]
    return prediction

# --- 3. Tampilan Streamlit ---

st.set_page_config(
    page_title="Prediksi Tingkat Risiko Depresi Mahasiswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Sistem Prediksi Tingkat Risiko Depresi Mahasiswa")
st.markdown("Aplikasi ini menggunakan model *Random Forest* untuk memprediksi tingkat risiko (Low, Medium, High) berdasarkan faktor akademis dan kepuasan belajar.")

st.sidebar.header("Input Data Mahasiswa")

# Input pengguna
academic_pressure = st.sidebar.slider(
    "1. Tekanan Akademik (0 = Rendah, 5 = Tinggi)",
    min_value=0, max_value=5, value=3
)

cgpa = st.sidebar.slider(
    "2. Rata-rata IPK (CGPA)",
    min_value=2.0, max_value=10.0, value=7.5, step=0.1
)

study_satisfaction = st.sidebar.slider(
    "3. Kepuasan Belajar (0 = Sangat Tidak Puas, 5 = Sangat Puas)",
    min_value=0, max_value=5, value=4
)

# Tombol prediksi
if st.sidebar.button("Prediksi Risiko"):
    if model is not None:
        # Kumpulkan input data
        input_data = {
            'Academic Pressure': academic_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction
        }

        # Lakukan prediksi
        risk_level_code = predict_risk(model, input_data)

        # Interpretasi hasil
        risk_map = {
            0: "Rendah (Low Risk)",
            1: "Sedang (Medium Risk)",
            2: "Tinggi (High Risk)"
        }

        color_map = {
            0: "green",
            1: "orange",
            2: "red"
        }

        risk_level = risk_map.get(risk_level_code, "Tidak Diketahui")
        color = color_map.get(risk_level_code, "grey")

        st.subheader("Hasil Prediksi")
        st.write(f"Berdasarkan data yang dimasukkan:")

        # Tampilkan hasil dengan warna
        st.markdown(
            f"Tingkat Risiko: <span style='color:{color}; font-size:24px; font-weight:bold;'>{risk_level}</span>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("Detail Input")
        st.json(input_data)

        st.subheader("Rekomendasi Cepat")
        if risk_level_code == 2:
            st.warning("⚠️ Risiko Tinggi! Segera cari bantuan profesional. Kurangi beban akademik, tingkatkan kualitas tidur, dan prioritaskan waktu istirahat.")
        elif risk_level_code == 1:
            st.info("💡 Risiko Sedang. Perhatikan keseimbangan hidup. Tingkatkan aktivitas sosial dan pastikan pola tidur dan makan teratur.")
        else:
            st.success("✅ Risiko Rendah. Pertahankan gaya hidup seimbang. Terus jaga kepuasan belajar dan kelola stres dengan baik.")

    else:
        st.error("Model prediksi belum siap.")
