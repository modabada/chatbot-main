import nltk


def splitter(title):
    import re

    l = len(title) + 4
    l += len(re.findall("[ㄱ-힣]", title))
    print()
    print("-" * l)
    print(" ", title, " ")
    print("-" * l)


# 문장 토큰화
splitter("문장 토큰화")
from nltk import sent_tokenize

text_sample = "natural Language Processing, or NLP, is the process of extracting the meaning, or intent, behind human language. In the field of Converstational artificial intelligence (AI), NLP allows machines and applications to understand the intent of human language inputs, and then generate appropriate response, resulting in a natural converstation flow."
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)


# 단어 토큰화
splitter("단어 토큰화")
from nltk import word_tokenize

sentence = "This book is for deep learning learners"
words = word_tokenize(sentence)
print(words)


# 아포스트로피가 포함된 문장에서 단어 토큰화
splitter("아포스트로피가 포함된 문장에서 단어 토큰화")
from nltk.tokenize import WordPunctTokenizer

sentence = "it`s nothing that you don`t already know except most people aren`t aware of how their inner world works"
words = WordPunctTokenizer().tokenize(sentence)
print(words)


# 한글 토큰화 예제
import csv
from konlpy.tag import Okt
from gensim.models import word2vec


# 다운로드 필요한 csv 파일
f = open(r"./data2/ratings_train.txt", "r", encoding="utf-8")
rdr = csv.reader(f, delimiter="\t")
rdw = list(rdr)
# 임시로 데이터 개수 줄임
rdw = rdw[:20]
f.close()


# 한글 형태소 분석기 호출
splitter("한글 형태소 분석기 호출")
twitter = Okt()
result = list()
for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True)
    r = list()
    for word in malist:
        if not word[1] in ["Josa", "Emoi", "Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)


# 형태소 저장
with open("./data2/NaverMovie.nlp", "w", encoding="utf-8") as fp:
    fp.write("\n".join(result))


# Word2Vec 모델 생성
mData = word2vec.LineSentence("./data2/NaverMovie.nlp")
mModel = word2vec.Word2Vec(
    mData,
    vector_size=200,
    window=10,
    hs=1,
    min_count=2,
    sg=1,
)
mModel.save("navberMove.model")


# 불용어 제거


# 불용어 제거
splitter("불용어 제거")
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")
from nltk.tokenize import word_tokenize

sample_text = "One of the first things that we ask ourselves is what are the pors and cons of any task we perform"
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [
    word for word in text_tokens if not word in stopwords.words("english")
]
print("불용어 제거 전:", text_tokens)
print("불용어 제거 후:", tokens_without_sw)


# 어간 추출


# 포터 알고리즘
splitter("포터 알고리즘")
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

print(stemmer.stem("obesses"), stemmer.stem("obssesed"))
print(stemmer.stem("standardizes"), stemmer.stem("standardization"))
print(stemmer.stem("national"), stemmer.stem("nation"))
print(stemmer.stem("absentness"), stemmer.stem("absently"))
print(stemmer.stem("tribalical"), stemmer.stem("tribalicalized"))


# 랭커스터 알고리즘
splitter("랭커스터 알고리즘")
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

print(stemmer.stem("obesses"), stemmer.stem("obssesed"))
print(stemmer.stem("standardizes"), stemmer.stem("standardization"))
print(stemmer.stem("national"), stemmer.stem("nation"))
print(stemmer.stem("absentness"), stemmer.stem("absently"))
print(stemmer.stem("tribalical"), stemmer.stem("tribalicalized"))


# 표제어 추출


# 표제어 추출
splitter("표제어 추출")
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
lemma = WordNetLemmatizer()

print(lemma.lemmatize("obesses"), lemma.lemmatize("obssesed"))
print(lemma.lemmatize("standardizes"), lemma.lemmatize("standardization"))
print(lemma.lemmatize("national"), lemma.lemmatize("nation"))
print(lemma.lemmatize("absentness"), lemma.lemmatize("absently"))
print(lemma.lemmatize("tribalical"), lemma.lemmatize("tribalicalized"))


# 품사 정보가 포함된 표제어 추출
splitter("품사 정보가 포함된 표제어 추출")
print(lemma.lemmatize("obesses", "v"), lemma.lemmatize("obssesed", "a"))
print(lemma.lemmatize("standardizes", "v"), lemma.lemmatize("standardization", "n"))
print(lemma.lemmatize("national", "a"), lemma.lemmatize("nation", "n"))
print(lemma.lemmatize("absentness", "n"), lemma.lemmatize("absently", "r"))
print(lemma.lemmatize("tribalical", "a"), lemma.lemmatize("tribalicalized", "v"))
