from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn

from my_utils import get_folds, print_and_log, make_dir
from my_utils_regression import train, evaluate

from Alexnet import AlexNet
from resnets import ResNet18
from Mobilenet import MobileNetV2
from datasets import EmbrapaP2Dataset

def get_imagent_alexNet():
    net = alexnet(pretrained=True)
    
    regressor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(10*6*256, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1, bias=True))
    net.classifier = regressor

    net.name = "ImagenetAlexNet"
    return net

def get_alexNet():
    net = AlexNet(1)
    
    regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*6*256, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1, bias=True))
    
    net._classifier = regressor

    return net

def get_resnet18():
    net = resnet18(pretrained=False)
    net.fc = nn.Linear(512, 1)
    net.name = "ResNet18"
    return net

def get_my_resnet():
    net = ResNet18(1)
    regressor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(12*7*512, 1000),
        nn.Linear(1000, 1))
    
    net._classifier = regressor
    return net

def get_mobilenetv2():
    net = MobileNetV2()
    return net

def save_predictions(indexes, predictions_list, csvfile) -> None:
    for predictions in predictions_list:
        pred = [p.item() for p in predictions]
        for index, pred in zip(indexes, predictions):
            csvfile.write(f"{index+1}, {pred.item()}\n")

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("model", type=str, 
                        help="suported models: alexnet or resnet18")
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    get_model = None
    if args.model == "alexnet":
        get_model = get_alexNet
    if args.model == "imagenetalexnet":
        get_model = get_alexNet
    elif args.model == "resnet":
        get_model = get_resnet18
    elif args.model == "myresnet":
        get_model = get_my_resnet
    elif args.model == "mobilenet":
        get_model = get_mobilenetv2

    folder = args.experiment_folder

    n = len(EmbrapaP2Dataset(args.dataset_folder))

    csv = open(folder+"predctions.csv", "w+")

    folds = get_folds(n, 10)
    folds_losses = []
    losses = dict()
    for k in range(10):
        mylogfile = folder + f"fold{k}.log"
        print_and_log((f"Fold #{k}",), mylogfile)


        train_indexes = []
        for n in range(10):
            if n != k:
                train_indexes += folds[n]
        test_indexes = folds[k]

        dltrain = torch.utils.data.DataLoader(EmbrapaP2Dataset(args.dataset_folder, train_indexes, augment=True), 
                                                shuffle=True, batch_size=128)
        dltest = torch.utils.data.DataLoader(EmbrapaP2Dataset(args.dataset_folder, test_indexes), 
                                                batch_size=128)

        model = get_model()
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        chkpt_folder = f"{folder}fold{k}/"
        make_dir(chkpt_folder)

        training_loss, test_loss = train(model, opt,  nn.MSELoss(), dltrain, dltest, 
                                         args.epochs, lr_schedular=None,
                                         cuda=True, logfile=mylogfile,
                                         checkpoints=[10, 30, 40, 50],
                                         checkpoints_folder=chkpt_folder)

        predictions, loss = evaluate(model, dltest, nn.MSELoss())
        folds_losses.append(loss)

        print_and_log((f"Test Loss: {loss}", "\n"), mylogfile)

        save_predictions(test_indexes, predictions, csv)
        
        x = np.linspace(1, len(training_loss), len(training_loss))
        fig, axs = plt.subplots(2)
        
        axs[0].set_title("Training Loss")
        axs[0].plot(x, training_loss, c='c')

        axs[1].set_title("Validation Loss")
        axs[1].plot(x, test_loss, c='m')
        
        plt.tight_layout()
        plt.savefig(folder+f"fold{k}-training-test-loss.png")

        losses[f"fold#{k}"] = training_loss

    csv.close()

    plt.figure()
    x = np.linspace(1, len(folds_losses), len(folds_losses))
    plt.plot(x, folds_losses)
    plt.savefig(folder+"folds-losses.png")

    losses["folds-losses"] = folds_losses

    with open(folder+f"losses.bin", "wb+") as fp:
        pk.dump(losses, fp)



