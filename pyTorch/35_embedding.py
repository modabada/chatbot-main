from __future__ import print_function
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import gensim
from gensim.models import Word2Vec, FastText, KeyedVectors


def splitter(title):
    import re

    l = len(title) + 4
    l += len(re.findall("[ㄱ-힣]", title))
    print()
    print("-" * l)
    print(" ", title, " ")
    print("-" * l)


# 원-핫 인코딩
splitter("원-핫 인코딩")
class2 = pd.read_csv("./data2/class2.csv")

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2["class2"])
print(train_x)


# 코퍼스에 카운터벡터 적용
splitter("카운터 벡터")
corpus = [
    "This is last chance.",
    "and if you do not have this chace",
    "you will never get any chance",
    "will you do get thsi one?",
    "please, get this chance",
]
vect = CountVectorizer()
vect.fit(corpus)
print(vect.vocabulary_)


# 배열 반환
splitter("카운터 벡터 결과를 배열로")
print(vect.transform(["you till never get any chance."]).toarray())


# 불용어를 제거한 카운터 벡터
splitter("불용어를 제거한 카운터벡터")
vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
print(vect.vocabulary_)


#  TF-IDF 를 적용한 후 행렬로 표현
splitter("TF-IDF 적용 후 행렬로 표현")
doc = ["I like machine learning", "I love deep learning", "I run everyday"]
tfidf_vectorize = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorize.fit_transform(doc)
doc_distance = tfidf_matrix * tfidf_matrix.T
print(
    "유사도를 위한",
    str(doc_distance.get_shape()[0]),
    "x",
    str(doc_distance.get_shape()[1]),
    "행렬을 만들었습니다",
)
print(doc_distance.toarray())


# 데이터셋을 메모리로 로딩하고 토큰화 적용
splitter("데이터셋을 메모리로 로딩하고 토큰화 적용")
warnings.filterwarnings(action="ignore")
sample = open("./data2/peter.txt", "r", encoding="utf8")
s = sample.read()
f = s.replace("\n", " ")
data = []
for i in sent_tokenize(f):
    tmp = list()
    for j in word_tokenize(i):
        tmp.append(j.lower())
    data.append(tmp)
print(data)
sample.close()


# 데이터셋에 CBOW 적용 후 'peter' 와 'wendy' 의 유사성 확인
splitter("CBOW 적용")
model1 = gensim.models.Word2Vec(
    data,
    min_count=1,
    vector_size=100,
    window=5,
    sg=0,
)
print(
    "Cosine similarity between 'peter' 'wendy' - CBOW: ",
    model1.wv.similarity("peter", "wendy"),
)
# peter 와 hook 유사성
print(
    "Cosine similarity between 'peter' 'hook' - CBOW: ",
    model1.wv.similarity("peter", "hook"),
)


# 데어터셋에 skip-gram 적용 후 peter 와 wendy 의 유사성 확인
splitter("skip-gram 적용")
model2 = gensim.models.Word2Vec(
    data,
    min_count=1,
    vector_size=100,
    window=5,
    sg=1,
)
print(
    "cosine similarity between 'peter' 'wendy' - Skip gram:",
    model2.wv.similarity("peter", "wendy"),
)
# peter 와 hook 유사성
print(
    "cosine similarity between 'peter' 'hook' - Skip gram:",
    model2.wv.similarity("peter", "hook"),
)


# FAstTExt
splitter("패스트텍스트")
model = FastText(
    "./data2/peter.txt",
    vector_size=4,
    window=3,
    min_count=1,
    epochs=10,
)

# peter, wendy  에 대한 코사인 유사도
sim_score = model.wv.similarity("peter", "wendy")
print(sim_score)
# peter, hook
sim_score = model.wv.similarity("peter", "hook")
print(sim_score)


splitter("한국어 처리")
# 다운로드 필요한 데이터
model_kr = KeyedVectors.load_word2vec_format("./data2/wiki/ko.vec")

# 유사도 확인
find_similar_to = "노력"
for similar_word in model_kr.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))


# 동물, 육식동물에는 긍정, 사람에는 부정적인 단어
similarities = model_kr.most_similar(positive=["동물", "육식동물"], negative=["사람"])
print(similarities)
