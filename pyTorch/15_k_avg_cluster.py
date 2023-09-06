import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


# 상품에 대한 연 지출 데이터 호출
data = pd.read_csv('./data/sales data.csv')

# 연속형 데이터와 명목형 데이터로 분류
# 명목
categorical_features = ['Channel', 'Region']
# 연속
continuos_features = ['Fresh', 'Milk' ,'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# 명목형 데이터는 분석을 위해 숫자로 변환
for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)


# 데이터 전처리 (스케일링 적용)
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.fit_transform(data)

# 적당한 k 값 추출
sum_of_squared_distances = list()
for k in range(1, 15):
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(range(1, 15), sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum of squared distances')
plt.title('optimal k')
plt.show()