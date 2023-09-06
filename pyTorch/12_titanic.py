import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/titanic/train.csv', index_col='PassengerId')
print(df.head())

# 데이터 전처리
# 생존 여부 예측을 위해 각 파라미터 사용
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
# 성별값을 0 또는 1 로 변환
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
#값이 없는 데이터 삭제
df = df.dropna()
x = df.drop('Survived', axis=1)
# Survived 를 예측 레이블로 사용
y = df['Survived']

# 훈련셋과 테스트셋으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# 결정 트리 모델 생성
model = tree.DecisionTreeClassifier()

# 모델 훈련
model.fit(x_train, y_train)

# 모델 예측
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict))

# 혼동 행렬로 성능측정
print(pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Pred not Survived', 'Pred Survived'],
    index=['not Survived', 'Survived']
))