import copy
import cv2
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import random
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import shutil
import time
import torch
import torchtext
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision.models as models
from torchvision.transforms import ToTensor
import tqdm.notebook as tqdm

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'

isCudaEnable = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if isCudaEnable else "cpu")


# 데이터 불러오기
data = pd.read_csv("./data/SBUX.csv")
print(data.dtypes)


# 인덱스 설정 및 데이터 타입 변경
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data["Volume"] = data["Volume"].astype(float)


# 훈련과 레이블 분리
x = data.iloc[:, :-1]
y = data.iloc[:, 5:6]
print(x)
print(y)


# 훈련과 테스트 데이터셋 정규화
ms = MinMaxScaler()
ss = StandardScaler()

x_ss = ss.fit_transform(x)
y_ms = ms.fit_transform(y)

x_train = x_ss[:200, :]
x_test = x_ss[200:, :]

y_train = y_ms[:200, :]
y_test = y_ms[200:, :]

print("Training Shape", x_train.shape, y_train.shape)
print("Testing Shape", x_test.shape, y_test.shape)


# 데이터셋 형태 변경
x_train_tensors = Variable(torch.Tensor(x_train))
x_test_tensors = Variable(torch.Tensor(x_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

x_train_tensors_f = torch.reshape(
    x_train_tensors,
    (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]),
)
x_test_tensors_f = torch.reshape(
    x_test_tensors,
    (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]),
)
print("Training Shape", x_train_tensors_f.shape, y_train_tensors.shape)
print("Testing Shape", x_test_tensors_f.shape, y_test_tensors.shape)


# GRU 모델의 네트워크
class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 은닉 상태에 대해 0으로 초기화하는 부분으로, LSTM 계층은 셀 상태가 있었지만 GRU 계층은 없음
        h_0 = Variable(
            torch.zeros(
                self.num_layers,
                x.size(0),
                self.hidden_size,
            )
        )
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


# 옵티마이저와 손실함수 지정
num_epochs = 1000
learning_rate = 0.0001

input_size = 5
hidden_size = 2
# GRU 계층의 개수
num_layers = 1

num_classes = 1
model = GRU(
    num_classes,
    input_size,
    hidden_size,
    num_layers,
    x_train_tensors_f.shape[1],
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 모델 학습
for epoch in range(num_epochs):
    outputs = model.forward(x_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# 그래프 출력을 위한 전처리
df_x_ss = ss.transform(data.iloc[:, :-1])
df_y_ms = ms.transform(data.iloc[:, -1:])

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))


# 모델 예측 결과 출력
train_predict = model(df_x_ss)
predicted = train_predict.data.numpy()
label_y = df_y_ms.data.numpy()

predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)
plt.figure(figsize=(10, 6))
plt.axvline(x=200, c="r", linestyle="--")

plt.plot(label_y, label="Actual Data")
plt.plot(predicted, label="Predicted Data")
plt.title("Time-Series Prediction")
plt.legend()
plt.show()
