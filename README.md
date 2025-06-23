# 🎧 Audio Speech Sentiment Analysis

Proyek ini merupakan implementasi sistem klasifikasi sentimen berbasis audio ucapan (speech) menggunakan Convolutional Neural Network (CNN). Proyek ini dibuat untuk penyelesaian Tugas Kelompok 2 Mata Kuliah Pengantar Pemrosesan Data Multimedia Program Studi Informatika Udayana tahun 2023.

Sistem menerima input berupa file audio format `.wav`, mengekstrak fitur secara manual (MFCC, ZCR, Spectral Centroid), kemudian memprediksi sentimen: **Positive**, **Neutral**, atau **Negative**. Aplikasi juga menyediakan antarmuka interaktif berbasis Streamlit.

---

## 🔧 Teknologi yang Digunakan

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Librosa

---

## 📦 Unduh Dataset & Model

📥 Dataset dan pre-trained model dapat diunduh di:  
👉 [Google Drive Link](https://drive.google.com/drive/folders/1RyNV2HDF5U-nKm4OuZUF5wMRkDT_yMvi?usp=sharing)

---

## 🧠 Cara Menjalankan

1. Pastikan Python dan library yang dibutuhkan telah terpasang.
2. Jalankan model training atau gunakan model yang sudah disediakan di folder `model_artifacts/`.
3. Jalankan aplikasi Streamlit: streamlit run app.py

---

## 👥 Tim Pengembang
Kelompok C4 - Informatika Udayana 2023:

I Putu Satria Dharma Wibawa (2308561045)
[🔗 SatriaDharma](github.com/SatriaDharma)

I Putu Andika Arsana Putra (2308561063)
[🔗 Andika Arsana](github.com/AndikaAP31)

Christian Valentino (2308561081)
[🔗 Christian Valentino](github.com/kriznoob)

Anak Agung Gede Angga Putra Wibawa (2308561099)
[🔗 Angga Wibawa](github.com/anggawww05)
