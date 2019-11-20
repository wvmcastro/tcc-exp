import torch
import torch.nn as nn

from BaseCNN import BaseConvNet

class BuildingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, s=1):
        super().__init__()
        self._conv1 = nn.Conv2d(in_dim, out_dim, 3, stride=s, padding=1)
        self._conv2 = nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1)
        self._relu = nn.ReLU()

        self._fix_dim = None
        if (in_dim != out_dim) or s != 1:
            self._fix_dim = nn.Conv2d(in_dim, out_dim, 1, stride=s)
    
    def forward(self, x):
        out = self._conv1(x)
        out = self._relu(out)
        out = self._conv2(out)

        if self._fix_dim is not None:
            x = self._fix_dim(x)

        out = self._relu(out + x)
        return out

class ResNet18(BaseConvNet):
    def __init__(self, nClasses: int):
        super().__init__("Resnet18")

        self._features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            BuildingBlock(64, 64),
            BuildingBlock(64, 64),
            BuildingBlock(64, 64),
            BuildingBlock(64, 128, s=2),
            BuildingBlock(128, 128),
            BuildingBlock(128, 128),
            BuildingBlock(128, 256, s=2),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 512, s=2),
            BuildingBlock(512, 512),
            BuildingBlock(512, 512),
            nn.AvgPool2d((1,1)),
        )

        self._classifier = nn.Sequential(
            nn.Linear(512, nClasses),
        )

    # def forward(self, x):
    #     print(x.shape)
    #     x = self._features(x)
    #     print(x.shape)
    #     x = self._classifier(x)
    #     return x