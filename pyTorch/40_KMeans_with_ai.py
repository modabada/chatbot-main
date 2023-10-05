import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from kmeans_pytorch import kmeans, kmeans_predict


df = pd.read_csv("./data/Iris.csv")
df.info()
print("-" * 40)
print(df)


# 워드 임베딩 (원-핫 인코딩)
data = pd.get_dummies(df, columns=["Species"])

# 데이터셋 분리
x, y = train_test_split(data, test_size=0.2, random_state=123)


# 연산 장치 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit(data).transform(x)
y_scaled = scaler.fit(y).transform(y)


# 데이터를 텐서로 변환
x = torch.from_numpy(x_scaled)
y = torch.from_numpy(y_scaled)


# 데이터셋에서 정답 제외
x = x[:, :5]
y = y[:, :5]


# 훈련 및 테스트 데이터셋 크기 출력
print(x.size())
print(y.size())
print(x)


# K 평균 군집화 적용
num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=x,
    num_clusters=num_clusters,
    distance="euclidean",
    device=device,
)


# 클러스터 id와 중심에 대한 값 확인
print(cluster_ids_x)
print(cluster_centers)


# K 평균 군집화 예측
cluster_ids_y = kmeans_predict(
    y,
    cluster_centers,
    "euclidean",
    device=device,
)


# 테스트셋에 대한 클러스터 id
print(cluster_ids_y)


# 예측 결과 그래프로 확인
plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(
    y[:, 0],
    y[:, 1],
    c=cluster_ids_y,
    cmap="viridis",
    marker="x",
)
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c="white",
    alpha=0.6,
    edgecolors="black",
    linewidths=2,
)
plt.tight_layout()
plt.show()
