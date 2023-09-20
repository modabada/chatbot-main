import random
import string
import time
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'

start = time.time()
TEXT = torchtext.data.Field(
    lower=True,
    fix_length=200,
    batch_first=False,
)
LABEL = torchtext.data.Field(sequential=False)


train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)


print(vars(train_data.examples[0]))


# 데이터 전처리
for example in train_data.examples:
    text = [x.lower() for x in vars(example)["text"]]
    text = [x.replace("<br", "") for x in text]
    text = ["".join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    vars(example)["text"] = text
print(vars(train_data.examples[0]))

# 데이터셋 분리
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)


print(f"Number of training examples:{len(train_data)}")
print(f"Number of validation examples:{len(valid_data)}")
print(f"Number of testing examples:{len(test_data)}")


# 단어 집합 생성
TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
print(LABEL.vocab.stoi)


BATCH_SIZE = [64 for _ in range(100)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embeding_dim = 100
hidden_size = 300

train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_sizes=BATCH_SIZE,
    device=device,
)


# 단어 임베딩 및 RNN 셀 정의
class RNNCell_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(RNNCell_Encoder, self).__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_size)

    def forward(self, inputs):
        bz = inputs.shape[1]
        ht = torch.zeros((bz, hidden_size)).to(device)
        for word in inputs:
            ht = self.rnn(word, ht)
            return ht


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)
        self.rnn = RNNCell_Encoder(embeding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 학습 함수 정의
def training(epoch, model, trainloader, validloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for b in trainloader:
        x, y = b.text, b.label
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    valid_correct = 0
    valid_total = 0
    valid_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in validloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            valid_correct += (y_pred == y).sum().item()
            valid_total += y.size(0)
            valid_running_loss += loss.item()

    epoch_valid_loss = valid_running_loss / len(validloader.dataset)
    epoch_valid_acc = valid_correct / valid_total

    print(
        "epoch: ",
        epoch,
        "loss: ",
        round(epoch_loss, 3),
        "accuracy:",
        round(epoch_acc, 3),
        "valid_loss: ",
        round(epoch_valid_loss, 3),
        "valid_accuracy:",
        round(epoch_valid_acc, 3),
    )
    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc


# 모델 학습
epochs = 5
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc = training(
        epoch,
        model,
        train_iter,
        valid_iter,
    )
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    valid_loss.append(epoch_valid_loss)
    valid_acc.append(epoch_valid_acc)

end = time.time()
print(end - start)


# 모델 예측 함수
def evaluate(epoch, model, testloader):
    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in testloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print(
        "epoch: ",
        epoch,
        "test_loss: ",
        round(epoch_test_loss, 3),
        "test_accuracy:",
        round(epoch_test_acc, 3),
    )
    return epoch_test_loss, epoch_test_acc


# 모델 예측 결과 확인
epochs = 5
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_test_loss, epoch_test_acc = evaluate(
        epoch,
        model,
        test_iter,
    )
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

end = time.time()
print(end - start)
