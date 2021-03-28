import torch
import torch.nn as nn

from torchvision.models.mobilenet import mobilenet_v2

class BaseConvNet(nn.Module):
    def __init__(self, name: str = "Model",
                       features: nn.Sequential = None, 
                       classifier: nn.Sequential = None):
        super().__init__()
        self.name = name
        self._features = features
        self._classifier = classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        print(x.shape)
        x = self._classifier(x)
        
        return x

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

#Models
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

class MobileNetV2(BaseConvNet):
    def __init__(self) -> None:
        net = mobilenet_v2(pretrained=False, num_classes=1)
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(12*8*1280, 1),
        )

        super().__init__("MobileNetV2", 
                         net.features, 
                         classifier)

class MaCNN(nn.Module):
    '''CNN from paper Ma2019'''
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
            nn.Linear(166656, 1), 
            nn.Linear(1,16)
        )

        self.classifier2 = nn.Sequential(
        	nn.Linear(48,1)
        )

    def forward(self,x):
        aux = x
        b,g,r = torch.chunk(x,3,1)
        r = self.features(r)
        r = torch.flatten(r, 1)
        r = self.classifier1(r)

        g = self.features(g)
        g = torch.flatten(g, 1)
        g = self.classifier1(g)

        b = self.features(b)
        b = torch.flatten(b, 1)
        b = self.classifier1(b)

        x = torch.cat((b,g,r), 1)
        x = self.classifier2(x)
        return x

#Darknet retirada do https://github.com/developer0hye/PyTorch-Darknet53
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(DarkResidualBlock(in_channels))
        return nn.Sequential(*layers)

def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)