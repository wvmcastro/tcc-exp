from torchvision.models.mobilenet import mobilenet_v2
import torch.nn as nn

from BaseCNN import BaseConvNet

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