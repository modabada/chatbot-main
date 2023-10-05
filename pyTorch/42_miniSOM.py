import numpy as np
from sklearn.datasets import load_digits
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone


# 숫자 필기 데이터셋 다운
digits = load_digits()
data = digits.data
label = digits.target


# 훈련 데이터셋을 MinSom 알고리즘에 적용
som = MiniSom(16, 16, 64, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
print("SOM 초기화")
som.train_random(data, 10000)
print("\nSOM 진행 종료")

bone()
pcolor(som.distance_map().T)
colorbar()


# 클래스에 대해 라벨 설정 및 색상 할당
label[label == "0"] = 0
label[label == "1"] = 1
label[label == "2"] = 2
label[label == "3"] = 3
label[label == "4"] = 4
label[label == "5"] = 5
label[label == "6"] = 6
label[label == "7"] = 7
label[label == "8"] = 8
label[label == "9"] = 9

markers = ["o", "v", "1", "3", "8", "s", "p", "x", "D", "*"]
colors = [
    "r",
    "g",
    "b",
    "y",
    "c",
    (0, 0.1, 0.8),
    (1, 0.5, 0),
    (1, 1, 0.3),
    "m",
    (0.4, 0.6, 0),
]
for cnt, xx in enumerate(data):
    w = som.winner(xx)
    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[label[cnt]],
        markerfacecolor="None",
        markeredgecolor=colors[label[cnt]],
        markersize=12,
        markeredgewidth=2,
    )
show()
