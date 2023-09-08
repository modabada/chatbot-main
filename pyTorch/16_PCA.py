import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


# 데이터 불러오기
x = pd.read_csv('./data/credit card.csv')
x = x.drop('CUST_ID', axis=1)
x.fillna(method='ffill', inplace=True)
# print(x.head())

# 데이터 전처리
scaler = StandardScaler()
# 평균이 0, 표준편차가 1 이 되도록 데이터 크기 조정
x_scaled = scaler.fit_transform(x)

# 데이터가 가우스 분포를 따르도록 정규화
x_normalized = normalize(x_scaled)
# 넘파이 배열을 데이터프레임 으로 변환
x_normalized = pd.DataFrame(x_normalized)

# 2차원으로 차원 축소 선언
pca = PCA(n_components=2)
x_principal = pca.fit_transform(x_normalized)
x_principal = pd.DataFrame(x_principal)
x_principal.columns = ['P1', 'P2']
# print(x_principal.head())

#DBSCAN 모델 생성 및 훈련
db_default = DBSCAN(eps=0.0375, min_samples=3).fit(x_principal)
# 각 데이터포인트에 할당된 모든 클러스터 레이블의 넘파이 배열을 저장
labels = db_default.labels_

# 결과 시각화
colors = ['y', 'g', 'b', 'k']

cvec = [colors[l]  for l in labels]
def graph_1():
    r = plt.scatter(x_principal['P1'], x_principal['P2'], color='y')
    g = plt.scatter(x_principal['P1'], x_principal['P2'], color='g')
    b = plt.scatter(x_principal['P1'], x_principal['P2'], color='b')
    k = plt.scatter(x_principal['P1'], x_principal['P2'], color='k')
    plt.figure(figsize=(9, 9))
    plt.scatter(x_principal['P1'], x_principal['P2'], c=cvec)
    plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))
    plt.show()
# graph_1()

# 모델 튜닝
db = DBSCAN(eps=0.0375, min_samples=50).fit(x_principal)
# min_samples 수를 변경해서 큰 값을 넣는다면 작은 규모의 클러스터가 무시된다

labels1 = db.labels_

colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
cvec = [colors1[l] for l in labels1]

def graph_2():
    plt_color = [
        plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=c)
        for c in colors1
    ]
    plt.figure(figsize=(9, 9))
    plt.scatter(x_principal['P1'], x_principal['P2'], c=cvec)
    plt.legend(
        plt_color,
        ['Label' + str(i) for i in range(7)],
        scatterpoints=1,
        loc='upper left',
        ncol=3,
        fontsize=8
    )
    plt.show()
graph_2()