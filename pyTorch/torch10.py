import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('./data/iris.data', names=names)

# 훈련용 셋과 테스트셋 분리
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
s = StandardScaler() # 특성 스케일링, 평균이 0, 표준편차가 1이 되도록 변환
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

# pickle 로 torch9 에서 생성한 모델 가져오기
with open ('./data/knn.pickle', 'rb') as f:
    knn = pickle.load(f)

# 모델 정확도
y_pred = knn.predict(x_test)
print('정확도: {}'.format(accuracy_score(y_test, y_pred)))