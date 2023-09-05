import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 


dataset = pd.read_csv('./data/car_evaluation.csv')
print(dataset.head())

fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams['figure.figsize'] = fig_size
dataset.output.value_counts().plot(
    kind='pie',
    autopct='%0.05f%%',
    colors=['lightblue', 'lightgreen', 'orange', 'pink'],
    explode=(0.05, 0.05, 0.05, 0.05)
)
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']
for category in categorical_columns:
    dataset[categorical_columns] = dataset[categorical_columns].astype('category')

price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)

# 배열을 텐서로 변환
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

# 레이블로 사용할 칼럼을 텐서로 변환
outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten() # 1 dimension

# 범주형 칼럼을 N차원으로 변환
categorical_column_size = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_size = [(col_size, min(50, (col_size + 1) // 2)) for col_size in categorical_column_size]

# 데이터셋 분리
total_records = 1728
test_records = int(total_records * .2) # 전체 데이터중 20% 를 테스트셋으로 사용

categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records: total_records]

# 각 데이터들의 길이 출력
print(len(categorical_train_data))
print(len(train_outputs))
print(len(categorical_test_data))
print(len(test_outputs))

# 모델 네트워크 생성
class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in embedding_size]
        )
        self.embedding_dropout = nn.Dropout(p)
        
        all_layers = list()
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
        
        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)
    
    def forward(self, x_categorical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1) # concat tensor
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x

# 모델 생성
model = Model(categorical_embedding_size, 4, [200, 100, 50])
print("model info:")
print(model)

# 모델 파라미터
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CPU/GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, '자원을 이용합니다')

# train
epochs = 500
aggregated_losses = list()
train_outputs = train_outputs.to(device=device, dtype=torch.int64)
print('train 시작')
for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data).to(device)
    single_loss = loss_func(y_pred, train_outputs)
    aggregated_losses.append(single_loss)
    
    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    
    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 테스트셋 이용 모델 예측
test_outputs = test_outputs.to(device=device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_func(y_val, test_outputs)
print(f'Loss: {loss:.8f}')

# 모델 결과 확인
# print(y_val[:5])

#가장 큰 값을 갖는 인덱스
y_val = np.argmax(y_val, axis=1)
# print(y_val[:5])


# 테스트셋으로 정확도 확인
def splitter(title: str):
    l = len(title)
    margin = 6
    print('=' * (l + margin))
    print(' ' * (margin // 2), title, sep='')
    print('=' * (l + margin))
splitter('confusion_matrix')
print(confusion_matrix(test_outputs, y_val))
splitter('classification_report')
print(classification_report(test_outputs, y_val))
splitter('accuracy_score')
print(accuracy_score(test_outputs, y_val))