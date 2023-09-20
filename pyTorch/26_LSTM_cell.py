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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
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
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if cuda:
    torch.cuda.manual_seed_all(125)


# 데이터 전처리
mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # 평균 0.5, 표준편차 1.0 으로 정규화
        transforms.Normalize((0.5,), (1.0,)),
    ]
)

# 데이터셋 받기
download_root = "../chap07/MNIST_DATASET"

train_dataset = MNIST(
    download_root,
    transform=mnist_transform,
    train=True,
    download=True,
)
valid_dataset = MNIST(
    download_root,
    transform=mnist_transform,
    train=False,
    download=True,
)
test_dataset = MNIST(
    download_root,
    transform=mnist_transform,
    train=False,
    download=True,
)


# 데이터셋을 메모리로 가져오기
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
valid_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)


# 변수 값 지정
batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)


# LSTM 셀
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)


# LSTM 모델
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        bias=True,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            )
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available():
            c0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            )
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), hidden_dim))

        outs = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
