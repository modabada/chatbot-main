import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('./data/iris.data', names=names)

# 훈련용 셋과 테스트셋 분리
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
s = StandardScaler() # 특성 스케일링, 평균이 0, 표준편차가 1이 되도록 변환
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

# 모델 생성 및 훈련
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train, y_train)

# 모델 정확도
y_pred = knn.predict(x_test)
print('정확도: {}'.format(accuracy_score(y_test, y_pred)))

# k 값 조정
k = 10
acc_array = np.zeros(k)
for k in np.arange(1, k + 1, 1):
    classifier = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k - 1] = acc
max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print(acc_list)
print('최적의 k는', k + 1, '로, 정확도', max_acc, '% 입니다')

# 입력받아서 테스트
def use_input():
    while True:
        a, b, c, d = map(float, input('input number: ').split(', '))
        x_test = np.array([a, b, c, d])
        x_test = s.fit_transform(x_test)
        y_pred = knn.predict(x_test)
        print(y_pred)
# use_input()

with open('./data/knn.pickle', 'wb') as f:
    pickle.dump(knn, f)
