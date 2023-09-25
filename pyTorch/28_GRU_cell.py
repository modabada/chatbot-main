import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
isCudaEnable = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if isCudaEnable else torch.FloatTensor

torch.manual_seed(125)
if isCudaEnable:
    torch.cuda.manual_seed_all(125)


# 데이터 전처리 정의
mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
    ]
)


# 데이터 다운ㄹ드 및 전처리 적용
download_root = "./data/MNIST_DATASET"
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


# 데이터셋 메모리로 가져오기
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
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


# GRU 셀 네트워크
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        # LSTM 셀에서는 x2h + h2h 지만, GRU 셀은 개별적인 상태를 유지
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        # 3개의 게이트(망각, 입력, new)를 위해 3개로 쪼갬
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        # 새로운 게이트는 탄젠트 활성화함수 적용
        newgate = F.tanh(i_n + (resetgate + h_n))

        hy = newgate + inputgate * (hidden - newgate)
        return hy


# 전반적인 네트워크 구조
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if isCudaEnable:
            h0 = Variable(
                torch.zeros(
                    self.layer_dim,
                    x.size(0),
                    self.hidden_dim,
                ).cuda()
            )
        else:
            h0 = Variable(
                torch.zeros(
                    self.layer_dim,
                    x.size(0),
                    self.hidden_dim,
                )
            )
        outs = list()

        # LSTM 셀에서는 셀 상태에 대해서도 정의했지만, GRU 셀은 사용 안함
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


# 옵티마이저와 손실함수 정의
input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

if isCudaEnable:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 모델 학습 및 성능 검증
seq_dim = 28
loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if isCudaEnable:
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if isCudaEnable:
            loss.cuda()

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in valid_loader:
                if isCudaEnable:
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                if isCudaEnable:
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print(
                "Iteration: {}. Loss: {}. Accuracy: {}".format(
                    iter, loss.item(), accuracy
                )
            )


# 테스트 데이터셋을 이용한 모델 예측
def evaluate(model, val_iter):
    corrects, total, total_loss = 0, 0, 0
    model.eval()
    for imgs, labels in val_iter:
        if isCudaEnable:
            imgs = Variable(imgs.view(-1, seq_dim, input_dim).cuda())
        else:
            imgs = Variable(imgs.view(-1, seq_dim, input_dim)).to(device)

        # 책에서 안알려주는 label 이 cpu tensor 인 오류 수정
        labels = labels.to(device)
        logit = model(imgs).to(device)
        # 모든 오차를 더하라는 sum 파라미터
        loss = F.cross_entropy(logit, labels, reduction="sum")
        _, predicted = torch.max(logit.data, 1)

        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()
    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy


# 모델 예측 결과
test_loss, test_acc = evaluate(model, test_loader)
print("Test Loss: %5.2f | Test Accuary: %5.2f" % (test_loss, test_acc))
