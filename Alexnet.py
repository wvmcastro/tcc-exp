import torch.nn as nn

from BaseCNN import BaseConvNet


class AlexNet(BaseConvNet):
    def __init__(self, nClasses):
        features = nn.Sequential(
            # first layer
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            # second layer
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            # third layer
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=False),
            # fourth layer
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=False),
            # fifth layer
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2))
        
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*256, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nClasses, bias=True)
        )

        super().__init__("AlexNet", features, classifier)