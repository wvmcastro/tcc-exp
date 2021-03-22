from argparse import ArgumentParser
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from datasets import DeepPhenoDataset
from models import AlexNet
from my_utils import train, test

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="dataset folder")
    args = parser.parse_args()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],)

    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),])

    dataset = DeepPhenoDataset(args.path, transform)
    
    n = len(dataset)
    ntrain = int(0.7*n)
    ntest = n - ntrain
    dstrain, dstest = torch.utils.data.random_split(dataset, (ntrain, ntest))
    dltrain = torch.utils.data.DataLoader(dstrain, shuffle=True, batch_size=512)
    dltest  = torch.utils.data.DataLoader(dstest, shuffle=True, batch_size=512)

    net = AlexNet(len(dataset.classes))

    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters())
    # opt = torch.optim.SGD(net.parameters(), 0.001, 0.9, weight_decay=1e-6)
    training_loss, training_acc = train(net, opt, loss, dltrain, args.epochs)

    x = np.linspace(0, len(training_loss), len(training_loss))

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x, training_loss)
    axs[0].set_title("Training Loss")

    axs[1].plot(x, training_acc)
    axs[1].set_title("Training Accuracy")

    test_acc, _ = test(net, dltest)

    print("Avg training acc: ", np.array(training_acc).mean())
    print("Avg test acc: ", np.array(test_acc).mean())

    plt.show()
