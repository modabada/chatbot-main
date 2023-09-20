import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms
import tqdm.notebook as tqdm

# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 이미지 데이터셋 전처리
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "validate": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# 이미지 데이터셋을 불러온 후, 룬련, 검증, 테스트 로 분리
cat_dir = r"./data/dogs_vs_cat/Cat/"
dog_dir = r"./data/dogs_vs_cat/Dog/"

cat_images_filepaths = sorted([os.path.join(cat_dir, f) for f in os.listdir(cat_dir)])
dog_images_filepaths = sorted([os.path.join(dog_dir, f) for f in os.listdir(dog_dir)])
images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

random.seed(42)
random.shuffle(correct_images_filepaths)

# 훈련용 0~399 번째 이미지
train_images_filepath = correct_images_filepaths[:400]
# 검증용 400~뒤에서 10번째
validate_images_filepath = correct_images_filepaths[400:-10]
# 테스트용 마지막 10개
test_images_filepath = correct_images_filepaths[-10:]
print(
    len(train_images_filepath), len(validate_images_filepath), len(test_images_filepath)
)


# 테스트 데이터셋 이미지 확인 함수
def display_image_grid(images_filepaths, predicted_labels=tuple(), col=5):
    rows = len(images_filepaths) // col
    figure, ax = plt.subplots(nrows=rows, ncols=col, figsize=(12, 6))
    for idx, path in enumerate(images_filepaths):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(path).split(os.sep)[-2]
        pred_label = predicted_labels[idx] if predicted_labels else true_label
        color = "green" if true_label == pred_label else "red"
        ax.ravel()[idx].imshow(image)
        ax.ravel()[idx].set_title(pred_label, color=color)
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    plt.show()


# 위의 함수를 이용하여 이미지 표시
# display_image_grid(test_images_filepath)
# display_image_grid(train_images_filepath)
# display_image_grid(validate_images_filepath)


# 이미지 데이터셋 클래스 정의
class DogVSCatDataset(Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = img_path.split("/")[-1].split(".")[0]
        if label == "Dog":
            label = 1
        elif label == "Cat":
            label = 0
        return img_transformed, label


# 기본 변수 정의
size = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

# 데이터셋 정의
train_dataset = DogVSCatDataset(
    train_images_filepath, transform=ImageTransform(size, mean, std), phase="train"
)
validate_dataset = DogVSCatDataset(
    validate_images_filepath,
    transform=ImageTransform(size, mean, std),
    phase="validate",
)
test_dataset = DogVSCatDataset(
    validate_dataset, transform=ImageTransform(size, mean, std), phase="validate"
)
index = 0
print(train_dataset.__getitem__(index)[0].size())
print(validate_dataset.__getitem__(index)[1])


# 데이터로더 정의
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {"train": train_dataloader, "validate": validate_dataloader}

batch_iterator = iter(train_dataloader)
img, label = batch_iterator._next_data()
print(img.size())
print(label)


# 모델의 네트워크 클래스
class AlexNet(nn.Module):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 모델 객체 생성
model = AlexNet()
print(model)


# 옵티마이저와 손실함수 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# torchsummar 를 이용해 모델의 네트워크 구조 확인
summary(model, input_size=(3, 256, 256))


# 연산 장치 할당
model = model.to(device)
criterion = criterion.to(device)


# 모델 학습 함수
def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print("-" * 20)

        for phase in ["train", "validate"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for img, lab in tqdm.tqdm_notebook(dataloader_dict[phase]):
                img = img.to(device)
                # tqdm 라이브러리가 멍청이가 되버려서 수동으로 str to int 변환
                classes = {"cat": 0, "dog": 1}
                lab = torch.tensor([classes[e] for e in lab])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(img)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, lab)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * img.size(0)
                    epoch_corrects += torch.sum(preds == lab.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    return model


# 모델 학습
model = train_model(model, dataloader_dict, criterion, optimizer, 10)


# 모델 테스트
id_list = list()
pred_list = list()
_id = 0

with torch.no_grad():
    for test_path in tqdm.tqdm_notebook(test_images_filepath):
        img = Image.open(test_path)
        _id = test_path.split("/")[-1].split(".")[1]
        transform = ImageTransform(size, mean, std)
        img = transform(img, phase="validate")
        img = img.unsqueeze(0)
        img = img.to(device)

        model.eval()
        outputs = model(img)
        preds = F.softmax(outputs, dim=1)[:, 1].tolist()

        id_list.append(_id)
        pred_list.append(preds[0])

res = pd.DataFrame({"id": id_list, "label": pred_list})

res.sort_values(by="id", inplace=True)
res.reset_index(drop=True, inplace=True)

res.to_csv("./data/AlexNet.csv", index=False)


classes = {0: "cat", 1: "dog"}


# 테스트 데이터셋 이미지 출력 함수
def display_test_image_grid(images_filepaths, predicted_labels=tuple(), col=5):
    rows = len(images_filepaths) // col
    figure, ax = plt.subplots(nrows=rows, ncols=col, figsize=(12, 6))
    for idx, path in enumerate(images_filepaths):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = random.choice(res["id"].values)
        label = res.loc[res["id"] == a, "label"].values[0]
        print(label)
        if label > 0.5:
            label = 1
        else:
            label = 0

        true_label = os.path.normpath(path).split(os.sep)[-2].lower()
        pred_label = classes[label]
        color = "green" if true_label == pred_label else "red"

        ax.ravel()[idx].imshow(image)
        ax.ravel()[idx].set_title(pred_label, color=color)
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    plt.show()


# display_test_image_grid(test_images_filepath)
