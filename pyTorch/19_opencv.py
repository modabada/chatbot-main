import copy
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


# ! 이미지 경로에 이미지 추가 필요
data_path = './data/openCV/'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    #num_workers=8, # 멀티 프로세싱
    shuffle=True
)
print('train_dataset len:', len(train_dataset))

# 멀티프로세싱 시 .next() 이용
sample, label = iter(train_loader)._next_data()

def graph1():
    classes = {0: 'cat', 1: 'dog'}
    fig = plt.figure(figsize=(16, 24))
    for i in range(24):
        a = fig.add_subplot(4, 6, i + 1)
        a.set_title(classes[label[i].item()])
        a.axis('off')
        a.imshow(np.transpose(sample[i].numpy(), (1, 2, 0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
    plt.show()
# graph1()

resnet18 = models.resnet18(pretrained=True)
print(resnet18)