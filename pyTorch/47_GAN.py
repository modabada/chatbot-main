import imageio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pylab as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image


plt.style.use("ggplot")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 기본 변수 값
batch_size = 512
epochs = 200
sample_size = 64  # 생성자에 제공할 고정 크기의 노이즈 벡터에 대한 크기
nz = 128  # 잠재 백터의 크기, 생성자의 입력 크기와 동일해야 함
k = 1  # 판별자에 적용할 스텝 수


# MNIST 다운받은 후 정규화
tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="./data2",
    train=True,
    transform=tf,
    download=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=4,
)


# 생성자 네트워크
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


# 판별자 네트워크 생성
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)


# 생성자와 판별자 객체 생성
gen = Generator(nz).to(device)
disc = Discriminator().to(device)
print(gen)
print(disc)


# 옵티마이저와 손실함수 정의
optim_g = optim.Adam(gen.parameters(), lr=2e-4)
optim_d = optim.Adam(disc.parameters(), lr=2e-4)

criterion = nn.BCELoss()

losses_g = list()
losses_d = list()
images = list()


# 생성된 이미지 저장 함수 정의
def save_generator_image(image, path):
    save_image(image, path)


# 판별자 학습 함수
def train_disc(optimizer: torch.optim.Optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = torch.ones(b_size, 1).to(device)
    fake_label = torch.zeros(b_size, 1).to(device)
    optimizer.zero_grad()

    output_real = disc(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = disc(data_fake)
    loss_fake = criterion(output_fake, fake_label)

    loss_real.backward()
    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake


# 생성자 학습 함수
def train_gen(optimizer: torch.optim.Optimizer, data_fake):
    b_size = data_fake.size(0)
    real_label = torch.ones(b_size, 1).to(device)
    optimizer.zero_grad()
    output = disc(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss


# 모델 학습
gen.train()
disc.train()

for epoch in range(epochs):
    loss_g = 0
    loss_d = 0

    for idx, data in tqdm(
        enumerate(train_loader), total=len(train_dataset) // train_loader.batch_size
    ):
        image, _ = data
        image = image.to(device)
        b_size = len(image)

        for step in range(k):
            data_fake = gen(torch.randn(b_size, nz).to(device)).detach()
            data_real = image
            loss_d += train_disc(optim_d, data_real, data_fake)
        data_fake = gen(torch.randn(b_size, nz).to(device))
        loss_g += train_gen(optim_g, data_fake)

    gen_img = gen(torch.randn(b_size, nz).to(device)).cpu().detach()
    gen_img = make_grid(gen_img)

    save_generator_image(gen_img, f"./data2/gen_img/gen_img{epoch}.png")
    images.append(gen_img)
    epoch_loss_g = loss_g / idx
    epoch_loss_d = loss_d / idx
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    print(f"Epoch {epoch} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")


# 생성자와 판별자의 오차 확인
plt.figure()
losses_g = [f1.item() for f1 in losses_g]
plt.plot(losses_g, label="generator loss")
losses_d = [f2.item() for f2 in losses_d]
plt.plot(losses_d, label="Discriminator loss")
plt.legend()
plt.show()


# 생성된 이미지 출력
fake_images = gen(torch.randn(b_size, nz).to(device))
for i in range(10):
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (28, 28))
    plt.imshow(fake_images_img, cmap="gray")
    plt.savefig("./data2/gen_img/fake_images_img" + str(i) + ".png")
    plt.show()
