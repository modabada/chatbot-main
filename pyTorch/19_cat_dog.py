import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'


# ! 이미지 경로에 이미지 추가 필요
data_path = './data/cat_dog/'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.ImageFolder(data_path + 'train/', transform=transform)
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

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# 사전 훈련된 모델은 학습(가중치 조정)을 진행하지 않음
set_parameter_requires_grad(resnet18)

# 완전연결층 추가
resnet18.fc = nn.Linear(512, 2)

for name, param in resnet18.named_parameters():
    if param.requires_grad:
        print(name, param.data)


def tmp_model():
    # 모델 객체 생성 및 손실함수 정의
    tmp_model = models.resnet18(pretrained=True)


    # 모델 합성곱층 가중치 고정
    for param in tmp_model.parameters():
        param.requires_grad = False

    # FC층은 학습 설정
    tmp_model.fc = nn.Linear(512, 2)
    for p in tmp_model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(tmp_model.fc.parameters())
    cost = torch.nn.CrossEntropyLoss()
    print(tmp_model)
# tmp_model()

# 모델 훈련 함수
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=13, is_train=True):
    since = time.time()
    acc_history = list()
    loss_history = list()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.to(device)
            # 기울기 0
            optimizer.zero_grad()
            # 순전파
            output = model(inputs)
            loss = criterion(output, labels)
            _, pred = torch.max(output, 1)
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 출력 결과와 레이블의 오차를 계산한 결과를 누적하여 저장
            running_loss += loss.item() * inputs.size(0)
            # 출력 결과와 레이블이 동일한지 확인한 결과를 누적하여 저장
            running_corrects += torch.sum(pred==labels.data)
        
        # 평균 오차 및 정확도 계산
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
        
        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        # 모델 재사용을 위해 저장
        torch.save(
            model.state_dict(), 
            os.path.join(data_path + '{0:0=2d}.pth'.format(epoch))
        )
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Bes Acc: {:4f}'.format(best_acc))
    return acc_history, loss_history


# 파라미터 학습 결과를 옵티마이저에 전달
params_to_update = []
for name,  param in resnet18.named_parameters():
    if param.requires_grad is True:
        params_to_update.append(param)
        print('\t', name)

optimizer = torch.optim.Adam(params_to_update)

# 학습
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
train_acc_hist, train_loss_hist = train_model(resnet18, train_loader, criterion, optimizer, device)

print(train_acc_hist)
print(train_loss_hist)

# 테스트 데이터 호출 및 전처리
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(data_path + 'test', transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    #num_workers=8, # 멀티 프로세싱
    shuffle=True
)
print(len(test_dataset))

# 테스트 데이터 평가 함수 생성
def eval_model(model, dataloaders, device):
    since = time.time()
    acc_history = list()
    best_acc = 0.0
    
    saved_models = glob.glob(data_path + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)
    
    for model_path in saved_models:
        print('Loading model', model_path)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        running_corrects = 0
        
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # autograd 를 사용하지 않겠다
            with torch.no_grad():
                outputs = model(inputs)
            
            _, pred = torch.max(outputs.data, 1)

            pred[pred >= 0.5] = 1 # 올바르게 예측
            pred[pred < 0.5] = 0 # 틀리게 예측
            running_corrects += pred.eq(labels.cpu()).int().sum()
        
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print('Acc: {:.4f}'.format(epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
        acc_history.append(epoch_acc.item())
        print()
    time_elapsed = time.time() - since
    print('Validation complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    
    return acc_history



# 테스트셋 사용
val_acc_hist = eval_model(resnet18, test_loader, device)


# 결과 시각화
def graph2():
    plt.plot(train_acc_hist)
    plt.plot(val_acc_hist)
    plt.show()
# graph2()


# 예측 이미지 출력을 위한 전처리 함수
def img_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * (np.array((.5, .5, .5)) + np.array((.5, .5, .5)))
    image = image.clip(0, 1)
    return image


# 예측 결과 출력
classes = {0: 'cat', 1: 'dog'}

images, labels  = iter(test_loader)._next_data()
output = resnet18(images)
_, pred = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))
for i in np.arange(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
    plt.imshow(img_convert(images[i]))
    ax.set_title(classes[labels[i].item()])
plt.subplots_adjust(bottom=-.2, top=0.6, hspace=0)
plt.show()