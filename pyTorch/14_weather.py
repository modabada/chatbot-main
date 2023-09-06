import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


dataset = pd.read_csv('./data/weather.csv')

# 데이터간 관계 시각화
def graph_actual():
    dataset.plot(x='MinTemp', y='MaxTemp', style='o')
    plt.title('min-max temperature')
    plt.xlabel('Mintemp')
    plt.ylabel('Maxtemp')
    plt.show()
# graph_actual()

# 데이터를 돌깁 변수와 종속 변수로 분리
x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 선형 회귀 모델 생성
regressor = LinearRegression()

# 훈련
regressor.fit(x_train, y_train)

# 모델 예측
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actural': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# 테스트셋으로 회귀선 표현
def graph_testset():
    plt.scatter(x_test, y_test, color='gray')
    plt.plot(x_test, y_pred, color='red', linewidth=2)
    plt.show()
graph_testset()

# 모델 평가
print('평균제곱법:', metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱법:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))