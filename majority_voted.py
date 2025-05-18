import os
import pandas as pd
from collections import Counter

# 다수결 함수 정의
def majority_vote_label(row):
    emotions = [row['1번 감정'], row['2번 감정'], row['3번 감정'], row['4번 감정'], row['5번 감정']]
    emotion_counts = Counter(emotions)
    return emotion_counts.most_common(1)[0][0]  # 가장 많은 감정 반환

# 처리할 파일 목록
base_path = "C:/TOY_TechOurYouth"
file_names = ['four.csv', 'five.csv', 'five_2.csv']
dataframes = []

# 각 파일 로드 및 source 정보 추가
for file in file_names:
    path = os.path.join(base_path, file)
    df = pd.read_csv(path, encoding='cp949')
    df["source"] = os.path.splitext(file)[0]  # 'four', 'five', 'five_2' 등
    dataframes.append(df)

# 모든 파일 병합
full_df = pd.concat(dataframes, ignore_index=True)

# 다수결 감정 라벨 추가
full_df['label'] = full_df.apply(majority_vote_label, axis=1)

# 필요한 열만 추출 후 저장
result_df = full_df[['wav_id', 'label', 'source']]
result_df.to_csv("C:/TOY_TechOurYouth/majority_voted_emotions.csv", index=False, encoding='utf-8-sig')

# 확인용 출력
print(result_df.head())