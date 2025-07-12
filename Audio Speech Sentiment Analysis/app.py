import streamlit as st
import os
import tempfile

from artifacts_and_predict import load_artifacts, predict

MODEL_ARTIFACTS_PATH = 'model_artifacts'

# Load artefak
@st.cache_resource
def get_model_artifacts():
    return load_artifacts(MODEL_ARTIFACTS_PATH)

model, scaler, encoder = get_model_artifacts()

# Streamlit UI
st.set_page_config(page_title="Audio Speech Sentiment Analysis", layout="centered")
st.title("🗣️ Audio Speech Sentiment Analysis")
st.markdown("Unggah file audio (`.wav`), pilih label aslinya, lalu klik tombol prediksi untuk mengetahui hasil model.")

if model and scaler and encoder:
    st.success("Model berhasil dimuat dan siap digunakan!")

    uploaded_file = st.file_uploader("📁 Unggah file audio (.wav)", type=["wav"])
    true_label = st.selectbox("🏷️ Pilih label asli:", ["Positive", "Neutral", "Negative"])
    predict_button = st.button("🔍 Prediksi Sentimen")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

    if predict_button:
        if uploaded_file is None:
            st.warning("Mohon unggah file audio terlebih dahulu.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_audio_path = tmp_file.name

            st.info(f"Memproses file: {uploaded_file.name}")

            try:
                predicted_sentiment, confidence = predict(
                    temp_audio_path, model, scaler, encoder
                )
                if predicted_sentiment is not None:
                    st.subheader("📊 Hasil Prediksi")
                    if predicted_sentiment == "Positive":
                        st.success("**Sentimen Terprediksi: Positif** 😊")
                    elif predicted_sentiment == "Neutral":
                        st.info("**Sentimen Terprediksi: Netral** 😐")
                    else:
                        st.error("**Sentimen Terprediksi: Negatif** 😠")

                    st.write(f"Label Asli: **{true_label}**")

                    if predicted_sentiment == true_label:
                        st.success("✅ Prediksi **benar**.")
                    else:
                        st.error("❌ Prediksi **salah**.")
                else:
                    st.warning("Gagal memproses audio. Silakan coba lagi.")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

else:
    st.error("Gagal memuat model. Pastikan folder `model_artifacts` berisi artefak model yang benar.")

# Footer
st.markdown("""<hr style="margin-top:50px;">""", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Dibuat oleh: Kelompok C4 Informatika Udayana Angkatan 2023<br>
        <br>
        I Putu Satria Dharma Wibawa (2308561045)<br>
        I Putu Andika Arsana Putra (2308561063)<br>
        Christian Valentino (2308561081<br>
        Anak Agung Gede Angga Putra Wibawa (2308561099)<br>
        <br>
        © 2025 - All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)