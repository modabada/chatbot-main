import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
# 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터셋 받기
train_dataset = torchvision.datasets.FashionMNIST('./data/CNN_test', download=True, train=True, transform=transform.Compose([transform.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST('./data/CNN_test', download=True, train=False, transform=transform.Compose([transform.ToTensor()]))

# 데이터로더에 데이터 전달
train_loader = DataLoader(train_dataset, batch_size=100)
test_loader = DataLoader(test_dataset, batch_size=100)

# 분류에 사용될 클래스 정의
labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Angle Boot'
}

def graph1():
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img_xy = np.random.random_integers(len(train_dataset))
        img = train_dataset[img_xy][0][0,:,:]
        fig.add_subplot(rows, columns, i)
        plt.title(labels_map[train_dataset[img_xy][1]])
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()
# graph1()

# 심층 신경망 모델 정의
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# 필요한 파라미터 정의
learning_rate = 0.001
model = FashionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(model)

# 학습
num_epochs = 5
count = 0
loss_list = list()
iteration_list = list()
accuracy_list = list()

prediction_list = list()
labels_list = list()

for epoch in range(num_epochs):
    for img, lab in train_loader:
        img, lab = img.to(device), lab.to(device)
        
        train = Variable(img.view(100, 1, 28, 28))
        lab = Variable(lab)
        
        outputs = model(train)
        loss = criterion(outputs, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        
        if count % 50 == 0:
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                prediction_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if count % 500 == 0:
            print('Iteration: {}, Loss: {}, Accuracy: {}%'.format(count, loss.data, accuracy))