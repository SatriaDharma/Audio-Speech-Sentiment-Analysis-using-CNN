import numpy as np
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

from extract_feature import extract_features

# Fungsi untuk load model dan artefak
def load_artifacts(model_path='model_artifacts'):
    try:
        model = load_model(os.path.join(model_path, 'audioSpeechSentimentAnalysis_model.h5'))
        with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_path, 'encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        with open(os.path.join(model_path, 'max_pad_len.pkl'), 'rb') as f:
            max_pad_len = pickle.load(f)
        return model, scaler, encoder, max_pad_len
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None, None

# Fungsi untuk memprediksi satu file audio
def predict(audio_filepath, model, scaler, encoder, max_pad_len):
    features = extract_features(audio_filepath, max_pad_len=max_pad_len)
    if features is None:
        return "Gagal mengekstrak fitur dari file audio.", None

    features_scaled = scaler.transform(features.reshape(1, -1))
    features_reshaped = features_scaled[..., np.newaxis]

    prediction_probs = model.predict(features_reshaped)
    predicted_class_index = np.argmax(prediction_probs)
    predicted_sentiment = encoder.inverse_transform([predicted_class_index])[0]
    confidence = prediction_probs[0][predicted_class_index]

    return predicted_sentiment, confidence