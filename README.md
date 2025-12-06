# Mental-health-# Aplikasi Prediksi Akademik dengan Streamlit

Aplikasi web untuk prediksi menggunakan model machine learning yang disimpan dalam format pickle (.pkl).

## Fitur

- ✅ Upload model dari file .pkl
- ✅ Input form dengan validasi:
  - **Academic Pressure**: Skala 0-5
  - **CGPA**: Nilai 2.0 - 10.0 (bisa desimal)
  - **Study Satisfaction**: Skala 0-5
- ✅ Preprocessing otomatis
- ✅ Prediksi real-time
- ✅ Tampilan probabilitas (jika model mendukung)

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

1. Pastikan file model (.pkl) sudah tersedia
2. Jalankan aplikasi:
```bash
streamlit run app.py
```

3. Buka browser di `http://localhost:8501`

## Cara Menggunakan

1. **Upload Model**: 
   - Di sidebar, upload file model (.pkl) atau
   - Centang "Gunakan model dari path lokal" dan masukkan path ke file model

2. **Isi Form Input**:
   - Academic Pressure: Geser slider antara 0-5
   - CGPA: Masukkan nilai antara 2.0 - 10.0
   - Study Satisfaction: Geser slider antara 0-5

3. **Klik Tombol "Prediksi"** untuk mendapatkan hasil

## Struktur Data

Aplikasi ini mengharapkan model yang dilatih dengan fitur:
- `Academic Pressure` (0-5)
- `CGPA` (2.0-10.0)
- `Study Satisfaction` (0-5)

**Catatan**: Kolom ID tidak digunakan dalam prediksi (sudah dihapus).

## Preprocessing

Preprocessing otomatis dilakukan:
- Academic Pressure: Dibatasi dalam range 0-5
- CGPA: Dibatasi dalam range 2.0-10.0
- Study Satisfaction: Dibatasi dalam range 0-5

## Requirements

Lihat `requirements.txt` untuk daftar lengkap dependencies.

