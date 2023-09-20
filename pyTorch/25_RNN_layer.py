import time
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'

start = time.time()


# 데이터셋 받기
TEXT = torchtext.data.Field(
    sequential=True,
    batch_first=True,
    lower=True,
)
LABEL = torchtext.data.Field(
    sequential=False,
    batch_first=True,
)

train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(split_ratio=0.8)


# 데이터 전처리
TEXT.build_vocab(
    train_data,
    max_size=10000,
    min_freq=10,
    vectors=None,
)
LABEL.build_vocab(train_data)

BATCH_SIZE = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 데이터셋 분리
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
)

# 변수
vocab_size = len(TEXT.vocab)
# 긍정 / 부정 2개
n_classes = 2


# RNN 계층 신경망
class BasicRNN(nn.Module):
    def __init__(
        self,
        n_layers,
        hidden_dim,
        n_vocab,
        embed_dim,
        n_classes,
        dropout_p=0.2,
    ):
        super(BasicRNN, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(
            embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True
        )
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.rnn(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


# 모델 생성
model = BasicRNN(
    n_layers=1,
    hidden_dim=256,
    n_vocab=vocab_size,
    embed_dim=128,
    n_classes=n_classes,
    dropout_p=0.5,
)
model.to(device)


# 손실함수와 옵티마이저
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 모델 학습 함수
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        if b % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    e,
                    b * len(x),
                    len(train_iter.dataset),
                    100.0 * b / len(train_iter),
                    loss.item(),
                )
            )


# 모델 평가 함수
def evaluate(model, val_iter):
    model.eval()
    corrects, total, total_loss = 0, 0, 0

    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        # 1~2 의 데이터를 0~1 로
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction="sum")
        total += y.size(0)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy


# 학습 및 평가
LR = 0.001
EPOCHS = 5
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, valid_iter)
    print(
        "[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f"
        % (e, val_loss, val_accuracy)
    )


# 테스트셋으로 모델 예측
test_loss, test_acc = evaluate(model, test_iter)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))
