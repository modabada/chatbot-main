import datetime
import os
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from tqdm.auto import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 데이터셋을 다운받은 후 텐서 변환
tf = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data2",
    train=True,
    transform=tf,
    download=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
    # num_workers=4,
    pin_memory=False,
)
test_dataset = datasets.MNIST(
    root="./data2",
    train=False,
    transform=tf,
    download=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    # num_workers=4,
)


# 인코더 네트워크 생성
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        h_ = self.relu(self.input1(x))
        h_ = self.relu(self.input2(h_))
        mean = self.mean(h_)  # 평균
        log_var = self.var(h_)  # 분산(log 를 취한 표준편차)
        return mean, log_var


# 디코더 네트워크
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden1 = nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.relu(self.hidden1(x))
        h = self.relu(self.hidden2(h))
        x_hat = torch.sigmoid(self.output(h))
        return x_hat


# 변형 오토인코더 네트워크
class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Model, self).__init__()
        self.en = encoder
        self.de = decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.en(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.de(z)
        return x_hat, mean, log_var


# 인코더와 디코더 객체 생성
x_dim = 784
hidden_dim = 400
latent_dim = 200
epochs = 30
batch_size = 100

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(encoder=encoder, decoder=decoder)


# 손실함수 정의
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD


# 옵티마이저
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 모델 학습 함수 정의
saved_loc = "./data2/tensorboard_browser/"
writer = SummaryWriter(saved_loc)

model.train()


def train(
    epoch: int, model: Model, train_loader: DataLoader, optimizer: optim.Optimizer
):
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        BCE, KLD = loss_function(x, x_hat, mean, log_var)
        loss = BCE + KLD

        writer.add_scalar(
            "Train/Reconstruction Error",
            BCE.item(),
            batch_idx + epoch * (len(train_loader.dataset) / batch_size),
        )
        writer.add_scalar(
            "Train/KL-Diveragence",
            KLD.item(),
            batch_idx + epoch * (len(train_loader.dataset) / batch_size),
        )
        writer.add_scalar(
            "Train/Total Loss",
            loss.item(),
            batch_idx + epoch * (len(train_loader.dataset) / batch_size),
        )

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]=t Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(x),
                )
            )
    print(
        "======> Epoch: {} Avg loss: {:.4f}".format(
            epoch,
            train_loss / len(train_loader.dataset),
        )
    )


# 모델 평가 함수
def test(epoch: int, model: Model, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)
            x_hat, mean, log_var = model(x)
            BCE, KLD = loss_function(x, x_hat, mean, log_var)
            loss = BCE + KLD

            writer.add_scalar(
                "Test/Reconstruction Error",
                BCE.item(),
                batch_idx + epoch * (len(test_loader.dataset) / batch_size),
            )
            writer.add_scalar(
                "Test/KL-Divergence",
                KLD.item(),
                batch_idx + epoch * (len(test_loader.dataset) / batch_size),
            )
            writer.add_scalar(
                "Test/Total Loss",
                loss.item(),
                batch_idx + epoch * (len(test_loader.dataset) / batch_size),
            )

            test_loss += loss.item()

            if batch_idx == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], x_hat.view(batch_size, x_dim)[:n]])
                grid = torchvision.utils.make_grid(comparison.cpu())
                writer.add_image(
                    "Test image - Above: real data, below: reconstruction data",
                    grid,
                    epoch,
                )


# 모델 학습
for epoch in tqdm(range(0, epochs)):
    train(epoch, model, train_loader, optimizer)
    test(epoch, model, test_loader)
    print("\n")
writer.close()
