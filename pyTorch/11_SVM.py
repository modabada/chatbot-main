from sklearn import svm, metrics, datasets, model_selection
# import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# iris 데이터 불러오고 훈련셋, 테스트셋 분리
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data,
    iris.target,
    test_size = 0.6,
    random_state = 42
)

# svm 모델에 대한 정확도
svm = svm.SVC(kernel='linear', C=1, gamma=0.5)
# 훈련 데이터로 svm 모델 훈련
svm.fit(x_train, y_train)
# 테스트데이터로 예측
predictions = svm.predict(x_test)
score = metrics.accuracy_score(y_test, predictions)
print('정확도: {0:f}'.format(score))