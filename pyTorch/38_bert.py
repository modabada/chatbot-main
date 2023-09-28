import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터셋 불러오기
train_df = pd.read_csv("./data2/ratings_train.txt", sep="\t")
valid_df = pd.read_csv("./data2/ratings_test.txt", sep="\t")
test_df = pd.read_csv("./data2/ratings_test.txt", sep="\t")


# 일부 데이터셋만 사용 (예제 진행중 속도)
train_df = train_df.sample(frac=0.1, random_state=500)
valid_df = valid_df.sample(frac=0.1, random_state=500)
test_df = test_df.sample(frac=0.1, random_state=500)


# 데이터셋 생성
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    __len__ = lambda self: len(self.df)

    def __getitem__(self, idx):
        txt = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return txt, label


# 데이터셋의 데이터를 데이터로더로 전달
train_dataset = CustomDataset(train_df)
valid_dataset = CustomDataset(valid_df)
test_dataset = CustomDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)


# 버트 토크나이저 다운로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.to(device)


# 최적화 모델 저장
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        print(f"Model save failed: path is None")
        return

    state_dict = {
        "moedel_state_dict": model.state_dict(),
        "valid_loss": valid_loss,
    }
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model):
    if load_path == None:
        print(f"Model load failed: path is None")
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict("valid_loss")


def save_matrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        print("Matric save failed: path is None")
        return
    state_sict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list,
    }
    torch.save(state_sict, save_path)
    print(f"Matric saved to ==> {save_path}")


def load_matrics(load_path):
    if load_path == None:
        print("Matrics load failed: path  is None")
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Matrics loaded from <== {load_path}")
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["global_steps_list"],
    )


# 모델 훈련 함수 정의
def train(
    model,
    optim,
    criterion=nn.BCELoss(),
    num_epochs=5,
    eval_every=len(train_loader) // 2,
    best_valid_loss=float("Inf"),
):
    total_correct = 0.0
    total_len = 0.0
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = list()
    valid_loss_list = list()
    global_steps_list = list()

    model.train()
    for epoch in range(num_epochs):
        for text, label in train_loader:
            optim.zero_grad()
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            running_loss += loss.item()
            loss.backward()
            optim.step()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for text, label in valid_loader:
                        encoded_list = [
                            tokenizer.encode(t, add_special_tokens=True) for t in text
                        ]
                        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
                        sample = torch.tensor(padded_list)
                        sample, label = sample.to(device), label.to(device)
                        labels = torch.tensor(label)
                        outputs = model(sample, labels=labels)
                        loss, logits = outputs
                        valid_running_loss += loss.item()

                    avg_train_loss = running_loss / eval_every
                    avg_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(avg_train_loss)
                    valid_loss_list.append(avg_valid_loss)
                    global_steps_list.append(global_step)

                    running_loss = 0
                    valid_running_loss = 0
                    model.train()

                    print(
                        "Epoch [{} / {}], Step [{} / {}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            global_step,
                            num_epochs * len(train_loader),
                            avg_train_loss,
                            avg_valid_loss,
                        ),
                    )

                    if best_valid_loss > avg_valid_loss:
                        best_valid_loss = avg_valid_loss
                        save_checkpoint("./data2/model.pt", model, best_valid_loss)
                        save_matrics(
                            "./data2/matricss.pt",
                            train_loss_list,
                            valid_loss_list,
                            global_steps_list,
                        )
    save_matrics(
        "./data2/matricss.pt",
        train_loss_list,
        valid_loss_list,
        global_steps_list,
    )
    print("훈련 종료")


# 모델의 파라미터(옵티마지어) 미세 조정 및 훈련
optimizer = optim.Adam(model.parameters(), lr=2e-5)
train(model=model, optim=optimizer)


# 오차 정보를 그래프로 확인
train_loss_list, valid_loss_list, global_steps_list = load_matrics("./data/matricss.pt")

plt.plot(global_steps_list, train_loss_list, label="Train")
plt.plot(global_steps_list, valid_loss_list, label="Valid")
plt.xlabel("Global Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()


# 모델 평가 함수 정의
def evaluate(model, test_loader):
    y_pred = list()
    y_label = list()

    model.eval()
    with torch.no_grad():
        for text, label in test_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            output = model(sample, labels=labels)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_label.extend(labels.tolist())

    print("Classification 결과: ")
    print(classification_report(y_label, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_label, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")
    ax.set_title("confution matrix")
    ax.set_xlabel("predicted labels")
    ax.set_ylabel("true labels")
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])


# 모델 평가
best_model = model.to(device)

load_checkpoint("./data2/model.pt", best_model)
evaluate(best_model, test_loader)
