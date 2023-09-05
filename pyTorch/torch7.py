import torch.nn as nn
import torch


class SLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layer = nn.Linear(
            in_features=inputs, 
            out_features=1
        )
        self.activation = nn.Sigmoid
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=30,
                kernel_size=5
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(
                in_features=30 * 5 * 5,
                out_features=10
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x

def main():
    model = MLP()
    print(list(model.children()))
    


if __name__ == '__main__':
    main()