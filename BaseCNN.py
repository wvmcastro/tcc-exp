import torch.nn as nn
import torch

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
        x = self._classifier(x)
        return x
