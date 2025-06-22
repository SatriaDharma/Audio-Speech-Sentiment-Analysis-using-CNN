import os
import librosa
import numpy as np
import pandas as pd

def extract_features(audio_path, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Ekstraksi fitur MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = np.mean(mfccs.T, axis=0)

        # Ekstraksi fitur Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr = np.mean(zcr.T, axis=0)

        # Ekstraksi fitur Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = np.mean(spectral_centroid.T, axis=0)

        # Gabungkan fitur
        features = np.hstack([mfccs, zcr, spectral_centroid])

        # Pad atau truncate fitur agar memiliki panjang yang seragam
        if len(features) < max_pad_len:
            pad_width = max_pad_len - len(features)
            features = np.pad(features, (0, pad_width), mode='constant')
        else:
            features = features[:max_pad_len]

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

    return features

def load_and_extract(base_dir, labels, n_mfcc=40, max_pad_len=174):
    data = []

    for label in labels:
        path = os.path.join(base_dir, label)
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' not found. Skipping label '{label}'.")
            continue
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                filepath = os.path.join(path, filename)
                features = extract_features(filepath, n_mfcc, max_pad_len)
                if features is not None:
                    data.append({'filepath': filepath, 'sentiment': label, 'features': features})

    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    return df