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

def get_myalexnet_pretrained():
    net = get_alexNet()
    imagenet_alexnet = get_imagent_alexNet()
    net._features = imagenet_alexnet.features
    net.name = "MyAlexNetPretrained"

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

def get_resnet18_pretrained():
    net = resnet18(pretrained=True)
    net.fc = nn.Linear(512, 1)
    net.name = "ResNet18Pretrained"
    return net

def get_mobilenetv2():
    net = MobileNetV2()
    return net

def save_predictions(indexes, predictions_list, csvfile) -> None:
    for predictions in predictions_list:
        pred = [p.item() for p in predictions]
        for index, pred in zip(indexes, predictions):
            csvfile.write(f"{index+1}, {pred.item()}\n")

def create_checkpoints_list(epochs_between_checkpoints, epochs):
    checkpoints_list = []
    epochs_executed = epochs_between_checkpoints
    while (epochs_executed <= epochs):
        checkpoints_list.append(epochs_executed)
        epochs_executed += epochs_between_checkpoints
    return checkpoints_list

def plot_average_validation_loss(losses, number_of_epochs, number_of_folds, filename):
    x = np.linspace(1, number_of_epochs, number_of_epochs)

    average_validation_loss = np.zeros(number_of_epochs)
    for fold in range(number_of_folds):
        average_validation_loss += np.array(losses[f'fold#{fold}']["validation_loss"])

    average_validation_loss /= number_of_folds

    plt.figure()
    plt.plot(x, average_validation_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.savefig(filename)

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("model", type=str, 
                        help="supported models: alexnet or resnet18")
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--augment", type=str, default="no", help="options: no, yes, super")
    parser.add_argument("--epochs_between_checkpoints", type=int, default=100)
    parser.add_argument("--cuda_device_number", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_device_number}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)

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
    elif args.model == "myalexnetpretrained":
        get_model = get_myalexnet_pretrained
    elif args.model == "resnet18pretrained":
        get_model = get_resnet18_pretrained

    folder = args.experiment_folder

    make_dir(folder)

    n = len(EmbrapaP2Dataset(args.dataset_folder))

    csv = open(folder+"predictions.csv", "w+")

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

        dltrain = torch.utils.data.DataLoader(EmbrapaP2Dataset(args.dataset_folder, train_indexes, augment=args.augment), 
                                                shuffle=True, batch_size=args.batch_size)
        dltest = torch.utils.data.DataLoader(EmbrapaP2Dataset(args.dataset_folder, test_indexes), 
                                                batch_size=args.batch_size)

        model = get_model()
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        chkpt_folder = f"{folder}fold{k}/"
        make_dir(chkpt_folder)

        checkpoints_list = create_checkpoints_list(args.epochs_between_checkpoints, args.epochs)

        training_loss, test_loss = train(model, opt,  nn.MSELoss(), dltrain, dltest, 
                                         args.epochs, lr_schedular=None,
                                         cuda=True, logfile=mylogfile,
                                         checkpoints=checkpoints_list,
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
        plt.savefig(folder+f"fold{k}-training-test-loss.pdf")

        losses[f"fold#{k}"] = {"training_loss": training_loss, "validation_loss": test_loss}

    plot_average_validation_loss(losses, args.epochs, 10, folder+"avg_validation_loss.pdf")

    csv.close()

    plt.figure()
    x = np.linspace(1, len(folds_losses), len(folds_losses))
    plt.plot(x, folds_losses)
    plt.savefig(folder+"folds-losses.pdf")

    losses["folds-losses"] = folds_losses

    with open(folder+f"losses.bin", "wb+") as fp:
        pk.dump(losses, fp)



