import os
import pandas as pd
import librosa
import numpy as np

# 실제 wav 파일들이 있는 디렉토리 경로
base_audio_dir = r"C:\Users\ailab\PycharmProjects\toy\five_2"

# CSV 불러오기
df = pd.read_csv(r"C:\Users\ailab\PycharmProjects\toy\majority_voted_emotions.csv")

# 전체 경로로 변환
df["file_path"] = df["wav_id"].apply(lambda x: os.path.join(base_audio_dir, x+ ".wav"))
file_paths = df["file_path"].values
labels = df["label"].values

#print(file_paths)
#print(labels)

def preprocess_wav(file_path, label, sr=16000, duration=3.0, n_fft=1024, hop_length=256, n_mels=80):
    # tf.Tensor → Python string (★ 핵심 수정 포인트)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")
    if isinstance(label, tf.Tensor):
        label = label.numpy().decode("utf-8")  # ★ 문자열이므로 decode 필요

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

'''# 전체 파일에 대해 처리 및 출력
for i in range(len(file_paths)):
    try:
        mel_spec, label = preprocess_wav(file_paths[i], labels[i])
        print(f"[{i}] 라벨: {label}, Mel 스펙트로그램 shape: {mel_spec.shape}")
    except Exception as e:
        print(f"[{i}] 에러 발생: {file_paths[i]} → {e}")'''

import tensorflow as tf

def tf_preprocess(file_path, label):

    spectrogram, label = tf.py_function(
        func=preprocess_wav,
        inp=[file_path, label],
        Tout=[tf.float32, tf.string] # 근데 모델 학습할 때는 tf.string -> tf.int64
    )
    spectrogram.set_shape([80, 188, 1])  # 예상 shape 지정 (고정 길이 기준)
    return spectrogram, label

import matplotlib.pyplot as plt

# TensorFlow용 데이터셋 만들기
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(tf_preprocess)
dataset = dataset.batch(1)


# 저장할 폴더 경로
save_dir = r"C:\Users\ailab\PycharmProjects\toy\mel_images"
os.makedirs(save_dir, exist_ok=True)


# 하나만 출력해서 확인
for spectrogram, label in dataset.take(1):
    spec_np = spectrogram.numpy()[0, :, :, 0]  # shape: (1, 80, 187, 1) → (80, 187)
    label_np = label.numpy()[0] if label.shape.rank > 0 else label.numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Mel-spectrogram (Label: {label_np})")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    # 파일 이름 설정 후 저장
    save_path = os.path.join(save_dir, f"happiness_mel.png")
    plt.savefig(save_path)
    plt.close()