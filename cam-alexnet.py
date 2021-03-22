import os
from argparse import ArgumentParser
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from embrapa_experiment import get_alexNet
from embrapa_experiment import get_myalexnet_pretrained
from models import BaseConvNet

class CamAlexNet(nn.Module):
    def __init__(self, pretrained_model: BaseConvNet):
        super().__init__()
        
        self._alexnet = pretrained_model

        self.features = self._alexnet._features[:-1]
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.classifier = self._alexnet._classifier

        self._gradients = None
    
    def activation_hook(self, grad) -> None:
        self._gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        _ = x.register_hook(self.activation_hook)

        x = self.max_pool(x)
        x = self.classifier(x)
        return x
    
    @property
    def gradient(self) -> torch.Tensor:
        return self._gradients
    
    def get_activations(self, x) -> torch.Tensor:
        return self.features(x)


def load_model(model: nn.Module, pth_file: str) -> None:
    checkpoint = torch.load(pth_file)
    model.load_state_dict(checkpoint["state_dict"])

def load_image(img_path: str) -> torch.Tensor:
    x = Image.open(img_path)
    t = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor()])
    x = t(x)

    x = x.unsqueeze(0)
    return x

def draw_heatmap(model: BaseConvNet, imgpath) -> np.ndarray:
    x = load_image(imgpath)
    pred = model(x.cuda())

    pred[0,0].backward()
    gradients = cam_net.gradient

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    conv_activations = cam_net.get_activations(x.cuda()).detach()
    
    for i in range(conv_activations.shape[1]):
        conv_activations[:, i, ...] *= pooled_gradients[i]
    
    
    heatmap = torch.mean(conv_activations, dim=1).squeeze(0)
    heatmap = heatmap.cpu().data.numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(imgpath)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heated_img = img + 0.3 * heatmap

    return heated_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pth_file", type=str, help="pth file with the\
        model state dict")
    parser.add_argument("srcdir", type=str, help="src directory with the images")
    parser.add_argument("dstdir", type=str, help="dst directory where the \
                                                  heatmaps will be saved")

    args = parser.parse_args()

    # alexnet = get_alexNet()
    alexnet = get_myalexnet_pretrained()
    load_model(alexnet, args.pth_file)
    cam_net = CamAlexNet(alexnet).cuda()
    cam_net.eval()
    
    for root, _, files in os.walk(args.srcdir):
        if files != []:
            for f in files:
                if ".png" in f:
                    heatmap = draw_heatmap(cam_net, root+f)
                    cv2.imwrite(args.dstdir+f, heatmap)
    
    # cv2.imwrite(args.image.split(".")[0]+"-heatmap.png", heated_img)
    # plt.savefig(args.image.split(".")[0]+"-heatmap-matrix.png")
    # plt.show()
