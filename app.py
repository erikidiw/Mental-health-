import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Akademik",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    """Memuat model dari file pickle"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Fungsi preprocessing
def preprocess_data(data):
    """
    Preprocessing data sesuai dengan requirements:
    - ID dihapus (tidak ada di input)
    - Academic Pressure: scale 0-5
    - CGPA: min-max 2-10 (bisa desimal)
    - Study Satisfaction: scale 0-5
    """
    # Pastikan data dalam format DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame([data])
    
    # Pastikan kolom yang diperlukan ada
    required_columns = ['Academic Pressure', 'CGPA', 'Study Satisfaction']
    
    # Validasi kolom
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Kolom yang hilang: {', '.join(missing_columns)}")
        return None
    
    # Normalisasi Academic Pressure (0-5)
    if 'Academic Pressure' in df.columns:
        df['Academic Pressure'] = df['Academic Pressure'].clip(0, 5)
    
    # Normalisasi CGPA (2-10)
    if 'CGPA' in df.columns:
        df['CGPA'] = df['CGPA'].clip(2.0, 10.0)
    
    # Normalisasi Study Satisfaction (0-5)
    if 'Study Satisfaction' in df.columns:
        df['Study Satisfaction'] = df['Study Satisfaction'].clip(0, 5)
    
    # Pastikan urutan kolom sesuai dengan yang diharapkan model
    df_processed = df[required_columns].copy()
    
    return df_processed

# Sidebar untuk upload model
st.sidebar.title("⚙️ Konfigurasi Model")
model_file = st.sidebar.file_uploader(
    "Upload Model File (.pkl)",
    type=['pkl'],
    help="Upload file model pickle yang akan digunakan untuk prediksi"
)

# Atau gunakan model dari path lokal
use_local_model = st.sidebar.checkbox("Gunakan model dari path lokal")
local_model_path = None
if use_local_model:
    local_model_path = st.sidebar.text_input(
        "Path ke file model (contoh: model.pkl)",
        value="model.pkl"
    )

# Tentukan model yang akan digunakan
model_path = None
if model_file is not None:
    # Simpan file yang diupload
    model_path = f"temp_{model_file.name}"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
elif use_local_model and local_model_path:
    if Path(local_model_path).exists():
        model_path = local_model_path
    else:
        st.sidebar.warning(f"File {local_model_path} tidak ditemukan")

# Load model jika path tersedia
model = None
if model_path:
    model = load_model(model_path)
    if model:
        st.sidebar.success("✅ Model berhasil dimuat!")

# Header utama
st.title("📊 Aplikasi Prediksi Akademik")
st.markdown("---")

# Informasi aplikasi
st.info("""
**Petunjuk Penggunaan:**
1. Upload file model (.pkl) di sidebar atau gunakan model dari path lokal
2. Isi form input di bawah ini
3. Klik tombol "Prediksi" untuk mendapatkan hasil prediksi

**Aturan Input:**
- **Academic Pressure**: Skala 0-5
- **CGPA**: Nilai antara 2.0 - 10.0 (bisa desimal)
- **Study Satisfaction**: Skala 0-5
""")

# Form input
st.header("📝 Form Input Data")

col1, col2 = st.columns(2)

with col1:
    academic_pressure = st.slider(
        "Academic Pressure",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        help="Tingkat tekanan akademik (0 = sangat rendah, 5 = sangat tinggi)"
    )
    
    cgpa = st.number_input(
        "CGPA",
        min_value=2.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
        help="Cumulative Grade Point Average (2.0 - 10.0)"
    )

with col2:
    study_satisfaction = st.slider(
        "Study Satisfaction",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Tingkat kepuasan belajar (0 = sangat tidak puas, 5 = sangat puas)"
    )

# Tampilkan ringkasan input
st.subheader(" Ringkasan Input")
input_summary = pd.DataFrame({
    'Fitur': ['Academic Pressure', 'CGPA', 'Study Satisfaction'],
    'Nilai': [academic_pressure, cgpa, study_satisfaction]
})
st.dataframe(input_summary, use_container_width=True, hide_index=True)

# Tombol prediksi
st.markdown("---")
predict_button = st.button(" Prediksi", type="primary", use_container_width=True)

# Proses prediksi
if predict_button:
    if model is None:
        st.error(" Model belum dimuat! Silakan upload model terlebih dahulu.")
    else:
        # Siapkan data input
        input_data = {
            'Academic Pressure': academic_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction
        }
        
        # Preprocessing
        with st.spinner(" Memproses data..."):
            processed_data = preprocess_data(input_data)
        
        if processed_data is not None:
            try:
                # Prediksi
                with st.spinner("🔮 Memprediksi..."):
                    prediction = model.predict(processed_data)
                    
                    # Jika model memiliki predict_proba, tampilkan probabilitas juga
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(processed_data)
                        prob_df = pd.DataFrame(
                            probabilities,
                            columns=[f'Kelas {i}' for i in range(len(probabilities[0]))]
                        )
                
                # Tampilkan hasil
                st.success(" Prediksi berhasil!")
                st.markdown("---")
                
                # Hasil prediksi utama
                st.subheader(" Hasil Prediksi")
                
                # Format hasil berdasarkan tipe prediksi
                if isinstance(prediction[0], (int, np.integer)):
                    st.metric("Prediksi", f"Kelas {prediction[0]}")
                elif isinstance(prediction[0], (float, np.floating)):
                    st.metric("Prediksi", f"{prediction[0]:.4f}")
                else:
                    st.metric("Prediksi", str(prediction[0]))
                
                # Tampilkan probabilitas jika ada
                if hasattr(model, 'predict_proba'):
                    st.subheader(" Probabilitas Prediksi")
                    st.bar_chart(prob_df.T)
                    st.dataframe(prob_df, use_container_width=True)
                
                # Tampilkan data yang diproses
                with st.expander(" Lihat Data yang Diproses"):
                    st.dataframe(processed_data, use_container_width=True)
                
            except Exception as e:
                st.error(f" Error saat prediksi: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Dibuat dengan Streamlit | Model Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

