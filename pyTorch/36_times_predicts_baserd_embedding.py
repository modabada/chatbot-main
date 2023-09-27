import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def splitter(title):
    import re

    l = len(title) + 4
    l += len(re.findall("[ㄱ-힣]", title))
    print()
    print("-" * l)
    print(" ", title, " ")
    print("-" * l)


plt.style.use("ggplot")
glove_file = datapath("./data2/glove.6B.100d.txt")
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)


splitter("유사도")
# bill 과 유사한 단어 리스트 반환
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.similar("bill")
model.similar("cherry")


# woman 과 king 이 유사하며 man ㅘ 관련이 없는 단어 목록
result = model.most_similar(positive=["woman", "king"], negative=["man"])
print("{}: {:.4f}".format(*result[0]))


# 'austrailia', 'beer', trance 와 관련이 있는 데이터
def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]


print(analogy("austrailia", "beer", "franch"))

# tall, tallest, long 기반으로 새 단어 유츄
print(analogy)("tall", "tallest", "long")


# 유사도가 낮은 단어 반환
print(model.doesnt_match("breakfast cereal dinner lunch".split()))
