import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. CSV 불러오기
df = pd.read_csv(r"C:\TOY_TechOurYouth\majority_voted_emotions.csv")

# 전체 파일 경로 생성 (source 컬럼 기준 폴더 지정)
df["file_path"] = df.apply(lambda row: os.path.join(
    r"C:\TOY_TechOurYouth", row["source"], row["wav_id"] + ".wav"), axis=1)

# 2. 존재하는 파일만 필터링 + 로그 출력
def file_exists(path):
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        return False
    return True

df = df[df["file_path"].apply(file_exists)].reset_index(drop=True)

# 3. 전처리 함수
def preprocess_wav(file_path, label, sr=16000, duration=1.0, n_fft=1024, hop_length=256, n_mels=80):
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")
    if isinstance(label, tf.Tensor):
        label = label.numpy().decode("utf-8")

    y, _ = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.expand_dims(log_mel, axis=-1).astype(np.float32)

    return log_mel, label

# 4. TensorFlow 래퍼 함수
def tf_preprocess(file_path, label):
    spectrogram, label = tf.py_function(
        func=preprocess_wav,
        inp=[file_path, label],
        Tout=[tf.float32, tf.string]
    )
    spectrogram.set_shape([80, 63, 1])
    return spectrogram, label

# 5. TensorFlow Dataset 생성
file_paths = df["file_path"].values
labels = df["label"].values

dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)

# 6. Mel-spectrogram 시각화 및 저장
save_dir = r"C:\TOY_TechOurYouth\mel_images"
os.makedirs(save_dir, exist_ok=True)

for i, (spectrogram, label) in enumerate(dataset.take(10)):
    spec_np = spectrogram.numpy()[0, :, :, 0]
    label_np = label.numpy()[0].decode("utf-8")

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

    print(f"✅ 저장 완료: {save_path}")