import os
from pydub import AudioSegment
from spleeter.separator import Separator
import tempfile
from pydub.utils import mediainfo

# 📁 경로 설정
BASE_DIR = r"C:\TOY_TechOurYouth"
INPUT_AUDIO = os.path.join(BASE_DIR, "Inside_Out2_Trailer.mp4")  # 또는 .mp3, .mp4
TEMP_WAV_PATH = os.path.join(BASE_DIR, "temp_input.wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "chunks")  # 분리 결과 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ ffmpeg 경로 수동 설정 (필요한 경우)
# AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"

# ✅ 파일 변환 함수 (mp3, mp4 → wav)
def convert_to_wav_if_needed(input_path, target_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        print("🎵 입력 파일은 WAV입니다. 변환 생략.")
        return input_path
    else:
        print(f"🔄 {ext.upper()} → WAV 변환 중...")
        audio = AudioSegment.from_file(input_path)
        audio.export(target_path, format="wav")
        print("✅ 변환 완료:", target_path)
        return target_path

# ✅ 분할 + Spleeter 처리 함수
def process_long_audio_in_chunks(input_wav_path, chunk_duration_ms=15 * 60 * 1000):
    # 디버깅용 전체 길이 출력
    info = mediainfo(input_wav_path)
    print(f"🕒 실제 오디오 길이: {float(info['duration']) / 60:.2f}분")

    full_audio = AudioSegment.from_wav(input_wav_path)
    total_length = len(full_audio)

    vocals_combined = AudioSegment.silent(duration=0)
    accomp_combined = AudioSegment.silent(duration=0)

    separator = Separator('spleeter:2stems')

    print(f"🎧 전체 길이: {total_length / 1000:.1f}초 → {chunk_duration_ms / 1000:.0f}초 단위로 분할")

    for i, start in enumerate(range(0, total_length, chunk_duration_ms)):
        end = min(start + chunk_duration_ms, total_length)
        chunk = full_audio[start:end]
        print(f"🔹 [{i+1}] {start / 1000:.0f}~{end / 1000:.0f}초 분리 중...")

        # 임시파일로 chunk 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")

            try:
                # Spleeter 분리
                separator.separate_to_file(chunk_path, tmpdir)

                # 분리 결과 경로 설정
                chunk_folder = os.path.join(tmpdir, f"chunk_{i}")
                vocals_file = os.path.join(chunk_folder, "vocals.wav")
                accomp_file = os.path.join(chunk_folder, "accompaniment.wav")

                # 오디오 로딩
                vocals_audio = AudioSegment.from_wav(vocals_file)
                accomp_audio = AudioSegment.from_wav(accomp_file)

                # 병합
                vocals_combined += vocals_audio
                accomp_combined += accomp_audio

            except Exception as e:
                print(f"❌ 오류 발생 (chunk {i+1}): {e}")
                continue

    # 전체 파일 저장
    vocals_combined.export(os.path.join(OUTPUT_DIR, "vocals_full.wav"), format="wav")
    accomp_combined.export(os.path.join(OUTPUT_DIR, "accompaniment_full.wav"), format="wav")
    print("✅ 전체 분리 완료 및 저장 완료 (15분씩 처리)")

# ✅ 실행
if __name__ == "__main__":
    print("🚀 시작...")
    actual_input_path = convert_to_wav_if_needed(INPUT_AUDIO, TEMP_WAV_PATH)
    process_long_audio_in_chunks(actual_input_path)

    # 임시 변환 파일 삭제
    if actual_input_path == TEMP_WAV_PATH and os.path.exists(TEMP_WAV_PATH):
        os.remove(TEMP_WAV_PATH)
        print("🧹 임시 wav 파일 삭제 완료.")
