import torch
import torch.nn as nn
import cv2
'''
CNN from paper Ma2019
'''
class MaCNN(nn.Module):
    def __init__(self,nclass = 1):
        super(MaCNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=0), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128,256,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(819*256, 1), 
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

'''
CNN late fusion from papaer "Modeling yield respons to crop 
management using convolution neural networks
Alexandre Barbosa 2020

Split channel from the image and pass each rgb channel in a different layer of CNN
'''

class LfCNN(nn.Module):
    def __init__(self,nclass = 1):
        super(LfCNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,stride=1,padding=0), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(16, 1), 
            nn.Linear(1,16)
        )

        self.classifier2 = nn.Sequential(
        	nn.Linear(16,1)
        )
    def forward(self,x):
        b,g,r = cv2.split(x)
        r = self.features(r)
        r = torch.flatten(r, 1)
        r = self.classifier1(r)

        g = self.features(g)
        g = torch.flatten(g, 1)
        g = self.classifier1(g)

        b = self.features(b)
        b = torch.flatten(b, 1)
        b = self.classifier1(b)

        x = cv2.merge((b,g,r))
        x = self.classifier2(x)
        return x
