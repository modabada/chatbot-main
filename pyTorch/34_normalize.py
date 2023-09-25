import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 데이터셋 경로 지정 및 훈련과 테스트 용도로 분리
df = pd.read_csv("./data2/diabetes.csv")
x = df[df.columns[:-1]]
y = df["Outcome"]

x = x.values
y = torch.tensor(y.values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


# 훈련과 테스트용 데이터 정규화
ms = MinMaxScaler()
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
# (?, 1) 의 형태, 즉 열의 수만 1로 고정
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)


# 커스텀 데이터셋 생성
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    __len__ = lambda self: self.len


# 데이터로더에 데이터 담기
train_data = CustomDataset(
    torch.FloatTensor(x_train),
    torch.FloatTensor(y_train),
)
test_data = CustomDataset(
    torch.FloatTensor(x_test),
    torch.FloatTensor(y_test),
)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)


# 네트워크 생성
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.l1 = nn.Linear(8, 64, bias=True)
        self.l2 = nn.Linear(64, 64, bias=True)
        self.l_out = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.l_out(x)
        return x


# 손실함수와 옵티마이저 지정
