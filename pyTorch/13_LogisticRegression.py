import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


digits = load_digits()
print('Image Data shape:', digits.data.shape)
print('label data shape:', digits.target.shape)

# 데이터셋 시각화
plt.figure(figsize=(20, 4))
for i, (image, label) in enumerate(zip(digits.data[:5], digits.target[:5])):
    plt.subplot(1, 5, i + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('trainning: %i\n' % label, fontsize=20)
# plt.show()

# 훈련셋과 테스트셋 분리
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# 로지스틱 회귀 모델 생성
logisticRegr = LogisticRegression()

# 훈련
logisticRegr.fit(x_train, y_train)

# 일부 데이터를 이용해 모델 예측
# logisticRegr.predict(x_test[0].reshape(1, -1))
# logisticRegr.predict(x_test[:10])

# 전체 테스트셋을 사용해 모델 예측
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('score: ', score)

# 혼동 행렬 시각화
# 혼동 행렬
cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 9))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='0.3f', 
    linewidths=0.5, 
    square=True, 
    cmap='Blues_r'
)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()