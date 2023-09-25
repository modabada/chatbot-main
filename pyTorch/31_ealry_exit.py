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


plt.style.use("ggplot")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
    [
        trainsforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# 데이터셋 가져오기
train_dataset = MNIST.ImageFolder(
    root=r"./data/Archive/train",
    transforms=train_transform,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
val_dataset = MNIST.ImageFolder(
    root=r"./data/Archive/test",
    transforms=val_transform,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)


# 모델 생성
def resnet50(pretrainded=True, requires_grad=False):
    model = (models.resnet50(progress=True, pretrainded=True),)
