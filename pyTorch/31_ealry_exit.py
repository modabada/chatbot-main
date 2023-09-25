import argparse
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
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
import torchvision.models as models
from torchvision.transforms import ToTensor
import tqdm

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


plt.style.use("ggplot")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
train_dataset = datasets.ImageFolder(
    root=r"./data/Archive/train",
    transform=train_transform,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
val_dataset = datasets.ImageFolder(
    root=r"./data/Archive/test",
    transform=val_transform,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)


# 모델 생성
def resnet50(pretrainded=True, requires_grad=False):
    model = (models.resnet50(progress=True, pretrainded=True),)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(2048, 2)
    return model


# 학습률 동적으로 감소
# learning rate scheduler
class LRScheduler:
    def __init__(
        self,
        optim,
        patience=5,
        min_lr=1e-6,
        factor=0.5,
    ):
        self.optim = optim
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# 조기 종료
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="./data/checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss


# 인수 값 지정
parser = argparse.ArgumentParser()
parser.add_argument("--lr-scheduler", dest="lr_scheduler", action="store_true")
parser.add_argument("--early-stopping", dest="early_stopping", action="store_true")
args = vars(parser.parse_args())


# 사전 훈련된 모델의 파라미터 확인
print(f"Computation device: {device}")
model = models.resnet50(pretrained=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters")


# 옵티마이저와 손실함수 지정
lr = 0.001
epochs = 3  # 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# 오차, 정확도 및 모델의 이름
loss_plot_name = "loss"
acc_plot_name = "accuracy"
model_name = "model"

# 오차, 정확도 및 모델의 이름에 대한 문자열
if args["lr_scheduler"]:
    print("INFO: Initializing learning rate schudler")
    lr_scheduler = LRScheduler(optimizer)
    loss_plot_name = "lrs_loss"
    acc_plot_name = "lrs_accuracy"
    model_name = "lrs_model"
if args["early_stopping"]:
    print("INFO: Initializing early stopping")
    early_stopping = EarlyStopping()
    loss_plot_name = "es_loss"
    acc_plot_name = "es_accuracy"
    model_name = "es_model"


# 모델 학습 함수
def training(model, train_dataloader, train_dataset, optimizer, criterion):
    print("Training")
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(
        enumerate(train_dataloader),
        total=len(train_dataset) // train_dataloader.batch_size,
    )
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100.0 * train_running_correct / total
    return train_loss, train_accuracy


# 모델 검증 함수
def validate(model, test_dataloader, val_dataset, criterion):
    print("Validating")
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(
        enumerate(test_dataloader),
        total=len(val_dataset) // test_dataloader.batch_size,
    )
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / counter
        val_accuracy = 100.0 * val_running_correct / total
        return val_loss, val_accuracy


# 모델 학습
train_loss, train_accuracy = list(), list()
val_loss, val_accuracy = list(), list()

start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = training(
        model,
        train_dataloader,
        train_dataset,
        optimizer,
        criterion,
    )
    val_epoch_loss, val_epoch_accuracy = validate(
        model,
        val_dataloader,
        val_dataset,
        criterion,
    )

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    if args["lr_scheduler"]:
        lr_scheduler(val_epoch_accuracy)
    if args["early_stopping"]:
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            break

    print(
        f"Train loss: {train_epoch_loss: .4f}, Train Acc: {train_epoch_accuracy: .2f}"
    )
    print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}")
end = time.time()
print(f"Training time: {(end - start) / 60:.3f} minutes")


# 결과 출력
print("Saving loss and accuracy plots...")
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color="green", label="train accuracy")
plt.plot(val_accuracy, color="blue", label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f"./data/img/{acc_plot_name}.png")
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color="orange", label="train lossw")
plt.plot(val_loss, color="red", label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("./data/img/{loss_plot_name}.png")
plt.show()

print("saving model...")
torch.save(model.state_dict(), f"./data/img/{model_name}.pth")
print("training complete")
