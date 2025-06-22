import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from extract_feature import load_and_extract
from augment_data import augment_and_extract

def cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(train_dir, labels, model_save_path='model_artifacts', n_mfcc=40, max_pad_len=174):
    os.makedirs(model_save_path, exist_ok=True)

    print("\nMemuat dan mengekstrak fitur dari data asli...")
    df_original = load_and_extract(train_dir, labels, n_mfcc, max_pad_len)
    if df_original.empty:
        print("Error: Tidak ada data ditemukan di direktori training.")
        return

    X_original = np.array(df_original['features'].tolist())
    y_original = df_original['sentiment'].values

    encoder = LabelEncoder()
    y_encoded_original = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)
    print(f"Jumlah kelas sentimen: {num_classes}")
    print(f"Label numerik: {encoder.classes_}")

    scaler = StandardScaler()
    X_scaled_original = scaler.fit_transform(X_original)

    print(f"Total data asli: {len(df_original)} file")

    print("\nMelakukan augmentasi data dan ekstraksi fitur...")
    augmented_df = augment_and_extract(df_original, max_pad_len=max_pad_len)
    print(f"Total data setelah augmentasi: {len(augmented_df)} file")

    X_augmented = np.array(augmented_df['features'].tolist())
    y_augmented = encoder.transform(augmented_df['sentiment'].values)

    X_augmented_scaled = scaler.transform(X_augmented)

    X_train, X_test, y_train, y_test = train_test_split(X_augmented_scaled, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Bentuk X_train: {X_train.shape}")
    print(f"Bentuk X_test: {X_test.shape}")

    input_shape = (X_train.shape[1], 1)
    model = cnn(input_shape, num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

    print("\nMemulai pelatihan model...")
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    # Plot hasil pelatihan
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_history.png'))
    plt.show()

    # Evaluasi model
    print("\n--- Evaluasi Model pada Data Uji ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy pada data uji: {accuracy*100:.2f}%")
    print(f"Loss pada data uji: {loss:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(model_save_path, 'confusion_matrix.png'))
    plt.show()

    model.save(os.path.join(model_save_path, 'audioSpeechSentimentAnalysis_model.h5'))
    with open(os.path.join(model_save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_save_path, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    
    with open(os.path.join(model_save_path, 'max_pad_len.pkl'), 'wb') as f:
        pickle.dump(max_pad_len, f)

    print(f"\nModel, scaler, dan encoder berhasil disimpan di: {model_save_path}")

if __name__ == '__main__':
    TRAIN_DIR = 'C:/Belajar Coding/PPDM/Audio Speech Sentiment Analysis/Dataset/TRAIN'
    SENTIMENT_LABELS = ['Positive', 'Neutral', 'Negative']
    
    temp_df = load_and_extract(TRAIN_DIR, SENTIMENT_LABELS, max_pad_len=1000)
    if not temp_df.empty:
        features_lengths = [len(f) for f in temp_df['features']]
        MAX_PAD_LENGTH = int(np.percentile(features_lengths, 95))
        if MAX_PAD_LENGTH < (40 + 2):
            MAX_PAD_LENGTH = (40 + 2) + 10
        print(f"Ditemukan Max Pad Length optimal: {MAX_PAD_LENGTH}")
    else:
        MAX_PAD_LENGTH = 174
        print(f"Data train tidak ditemukan. Menggunakan Max Pad Length default: {MAX_PAD_LENGTH}")
    
    train_and_evaluate(TRAIN_DIR, SENTIMENT_LABELS, max_pad_len=MAX_PAD_LENGTH)