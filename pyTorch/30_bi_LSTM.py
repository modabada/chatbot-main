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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 데이터 로드
data = pd.read_csv("./data/SBUX.csv")
print(data.dtypes)


# 데이터 전처리
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data["Volume"] = data["Volume"].astype(float)


# 데이터셋 분리
x = data.iloc[:, :-1]
y = data.iloc[:, 5:6]


# 트랜스폼 규칙 정의
ms = MinMaxScaler()
ss = StandardScaler()
x_ss = ss.fit_transform(x)
y_ms = ms.fit_transform(y)


# 테스트와 훈련셋 분리
x_train = x_ss[:200, :]
x_test = x_ss[200:, :]
y_train = y_ms[:200, :]
y_test = y_ms[200:, :]


# 트랜스폼 적용
x_train_tensors = Variable(torch.Tensor(x_train)).to(device)
x_test_tensors = Variable(torch.Tensor(x_test)).to(device)
y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
y_test_tensors = Variable(torch.Tensor(y_test)).to(device)
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


# 모델 네트워크
class biLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
    ):
        super(biLSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # 제일 중요한 차이인데 이거 한줄이 다임 ㅇㅇ
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size),
        )
        c_0 = Variable(
            torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size),
        )
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out


# 모델 학습
num_epochs = 1000
learning_rate = 0.0001

input_size = 5
hidden_size = 2
num_layers = 1

num_classes = 1
model = biLSTM(
    num_classes,
    input_size,
    hidden_size,
    num_layers,
    x_train_tensors_f.shape[1],
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model.forward(x_train_tensors_f)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train_tensors)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# 모델 평가 및 시각화
df_x_ss = ss.transform(data.iloc[:, :-1])
df_y_ms = ms.transform(data.iloc[:, -1:])

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))
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
