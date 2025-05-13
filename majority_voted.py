import pandas as pd
from collections import Counter

# CSV 파일 불러오기
df = pd.read_csv("C:/Users/ailab/PycharmProjects/toy/five_2.csv", encoding='cp949')

# 다수결 함수 정의
def majority_vote_label(row):
    emotions = [row['1번 감정'], row['2번 감정'], row['3번 감정'], row['4번 감정'], row['5번 감정']]
    emotion_counts = Counter(emotions)
    return emotion_counts.most_common(1)[0][0]  # 가장 많은 감정 반환

# 다수결 감정 컬럼 추가
df['label'] = df.apply(majority_vote_label, axis=1)

# 필요한 열만 추출해서 출력하거나 저장
print(df[['wav_id', 'label']])

df[['wav_id', 'label']].to_csv("majority_voted_emotions.csv", index=False, encoding='utf-8-sig')
