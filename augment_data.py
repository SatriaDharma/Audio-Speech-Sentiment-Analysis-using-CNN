import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

from extract_feature import extract_features

def augment_audio(audio_path, sr_orig, output_dir="temp_augmented_audio"):
    y, _ = librosa.load(audio_path, sr=sr_orig)

    augmented_waves_info = []
    os.makedirs(output_dir, exist_ok=True)

    # Augmentasi data penambahan Noise (white noise)
    noise_amplitude = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amplitude * np.random.normal(size=y.shape[0])
    temp_path_noise = os.path.join(output_dir, f"aug_noise_{os.path.basename(audio_path)}")
    sf.write(temp_path_noise, y_noise, sr_orig)
    augmented_waves_info.append((temp_path_noise, y_noise))

    # Augmentasi data pergeseran Pitch (semitone = -2 atau 2)
    y_pitch_down = librosa.effects.pitch_shift(y=y, sr=sr_orig, n_steps=-2)
    temp_path_pitch_down = os.path.join(output_dir, f"aug_pitch_down_{os.path.basename(audio_path)}")
    sf.write(temp_path_pitch_down, y_pitch_down, sr_orig)
    augmented_waves_info.append((temp_path_pitch_down, y_pitch_down))

    y_pitch_up = librosa.effects.pitch_shift(y=y, sr=sr_orig, n_steps=2)
    temp_path_pitch_up = os.path.join(output_dir, f"aug_pitch_up_{os.path.basename(audio_path)}")
    sf.write(temp_path_pitch_up, y_pitch_up, sr_orig)
    augmented_waves_info.append((temp_path_pitch_up, y_pitch_up))

    # Augmentasi data pergeseran Waktu (time stretching)
    rate = np.random.uniform(0.8, 1.2)
    y_stretch = librosa.effects.time_stretch(y=y, rate=rate)
    temp_path_stretch = os.path.join(output_dir, f"aug_stretch_{os.path.basename(audio_path)}")

    sf.write(temp_path_stretch, y_stretch, sr_orig)
    augmented_waves_info.append((temp_path_stretch, y_stretch))

    return augmented_waves_info

def augment_and_extract(df_original, max_pad_len, output_dir="temp_augmented_audio"):
    augmented_data = []

    # Tambahkan data asli
    for index, row in df_original.iterrows():
        augmented_data.append({'filepath': row['filepath'], 'sentiment': row['sentiment'], 'features': row['features']})

    for index, row in df_original.iterrows():
        audio_path = row['filepath']
        sentiment = row['sentiment']
        _, sr_orig = librosa.load(audio_path, sr=None)
        augmented_waves_info = augment_audio(audio_path, sr_orig, output_dir)
        for temp_audio_path, _ in augmented_waves_info:
            aug_features = extract_features(temp_audio_path, max_pad_len=max_pad_len)
            if aug_features is not None:
                augmented_data.append({'filepath': temp_audio_path, 'sentiment': sentiment, 'features': aug_features})
            os.remove(temp_audio_path)

    # Hapus direktori sementara jika kosong
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.dropna(inplace=True)
    return augmented_df