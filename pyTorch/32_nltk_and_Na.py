import nltk
import pandas as pd


def splitter(title):
    import re

    l = len(title) + 4
    l += len(re.findall("[ㄱ-힣]", title))
    print()
    print("-" * l)
    print(" ", title, " ")
    print("-" * l)


nltk.download()
text = nltk.word_tokenize("Is it possible distinguishing cats and dogs")
print(text)

nltk.download("averaged_perceptron_tagger")


print(nltk.pos_tag(text))


# 전처리


# 결측치를 확인할 데이터 호출
df = pd.read_csv("./data2/class2.csv")


# 데이터 개수가 많아 임시로 개수 10개로 줄임
# df_k = df.keys()
# new_df = dict()
# for k in df_k[:10]:
#     new_df[k] = df[k]
# df = new_df


# 결측치 개수 확인
splitter("결측치 개수")
print(df.isnull().sum())


# 결측치 비율
splitter("결측치 비율")
print(df.isnull().sum() / len(df))


# 결측치 삭제 (모든 행이 Nan 일때)
splitter("결측치 삭제(모든 행이 NaN 일때)")
df = df.dropna(how="all")
print(df)


# 결측치 삭제 (데이터가 하나라도 Nan 이면 행 삭제)
splitter("결측치 삭제 (하나라도 NaN일때)")
df1 = df.dropna()
print(df1)


# 결측치를 0으로 채우기
splitter("결측치를 0으로 채우기")
df2 = df.fillna(0)
print(df)


# 결측치를 평균값으로 채우기
splitter("결측치를  평균값으로 채우기")
df["x"].fillna(df["x"].mean(), inplace=True)
print(df)
