import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

# CSV 불러오기 (source 컬럼 포함되어 있어야 함)
df = pd.read_csv(r"C:\TOY_TechOurYouth\majority_voted_emotions.csv")

# 전체 파일 경로 생성 (source 컬럼을 사용해서 폴더 자동 매핑)
df["file_path"] = df.apply(lambda row: os.path.join(
    r"C:\TOY_TechOurYouth", row["source"], row["wav_id"] + ".wav"), axis=1)

file_paths = df["file_path"].values
labels = df["label"].values

# 전처리 함수 정의
def preprocess_wav(file_path, label, sr=16000, duration=3.0, n_fft=1024, hop_length=256, n_mels=80):
    # tf.Tensor → Python string (★ 핵심 수정 포인트)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")
    if isinstance(label, tf.Tensor):
        label = label.numpy().decode("utf-8") # ★ 문자열이므로 decode 필요

    # 파일 경로 존재 여부 확인 (디버깅용)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일 없음: {file_path}")

    y, _ = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.expand_dims(log_mel, axis=-1).astype(np.float32)

    return log_mel, label

# TensorFlow용 전처리 래퍼 함수
def tf_preprocess(file_path, label):
    spectrogram, label = tf.py_function(
        func=preprocess_wav,
        inp=[file_path, label],
        Tout=[tf.float32, tf.string]  # 근데 모델 학습할 때는 tf.string -> tf.int64
    )
    spectrogram.set_shape([80, 188, 1])  # 예상 shape 지정 (고정 길이 기준)
    return spectrogram, label

# TensorFlow 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(tf_preprocess)
dataset = dataset.batch(1)

# 저장 경로 설정
save_dir = r"C:\TOY_TechOurYouth\mel_images"
os.makedirs(save_dir, exist_ok=True)

# 예시 10개 Mel-spectrogram 저장
for i, (spectrogram, label) in enumerate(dataset.take(10)):
    spec_np = spectrogram.numpy()[0, :, :, 0]   # shape: (1, 80, 187, 1) → (80, 187)
    label_np = label.numpy()[0] if label.shape.rank > 0 else label.numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Mel-spectrogram (Label: {label_np})")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{i:03d}_{label_np}.png")
    plt.savefig(save_path)
    plt.close()