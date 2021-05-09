from argparse import ArgumentParser
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import time
import sys

from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18
from torchvision.models import vgg11_bn
from torchvision.models import resnext50_32x4d, resnext101_32x8d
import torch
import torch.nn as nn

from my_utils import get_folds, print_and_log, make_dir, save_info
from my_utils_regression import train, evaluate, get_metrics

from models import AlexNet, ResNet18, MobileNetV2, MaCNN, LfCNN, darknet53

from datasets import EmbrapaP2Dataset

def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("model", type=str, 
                        help="supported models: alexnet, resnet18 or vggnet11")
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--augment", type=str, default="no", help="options: no, yes, super")
    parser.add_argument("--epochs_between_checkpoints", type=int, default=100)
    parser.add_argument("--cuda_device_number", type=int, default=0)
    parser.add_argument('--only_one_fold', dest='only_one_fold', action='store_true')
    parser.set_defaults(only_one_fold=False)
    
    return parser

def get_darknet53():
    net = darknet53(1)
    net.name = "DarkNet53"
    return net

def get_MaCNN():
    net = MaCNN(1)
    net.name = "MaCNN"
    return net
    
def get_lfCnn():
    net = LfCNN(1)
    net.name = "lfcnn"
    return net

def get_imagenet_alexNet():
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
    imagenet_alexnet = get_imagenet_alexNet()
    net._features = imagenet_alexnet.features
    net.name = "MyAlexNetPretrained"

    return net

def get_resnet18():
    net = resnet18(pretrained=False)
    net.fc = nn.Linear(512, 1)
    net.name = "ResNet18"
    return net

def get_resnext50():
    net = resnext50_32x4d(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.name = "ResNext50"
    return net
    
def get_resnext101():
    net = resnext101_32x8d(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.name = "ResNext101"
    return net

def get_resnext50_pretrained():
    # eu nao apoio essa distinção de pretreinado ou não >:(
    net = resnext50_32x4d(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.name = "ResNext50Pretrained"
    return net
    
def get_resnext101_pretrained():
    net = resnext101_32x8d(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.name = "ResNext101Pretrained"
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

def get_vggnet11():
    net = vgg11_bn(pretrained=False)
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1000),
        nn.Linear(1000, 1))
    net.name = "VGGNet11"
    return net

def get_vggnet11_pretrained():
    net = vgg11_bn(pretrained=True)
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1000),
        nn.Linear(1000, 1))
    net.name = "VGGNet11Pretrained"
    return net

def save_predictions(indexes, predictions_list, csvfile) -> None:
    for predictions in predictions_list:
        pred = [p.item() for p in predictions]
        for index, pred in zip(indexes, predictions):
            # TODO: Esse +1 deve servir para algo!!!!!!
            csvfile.write(f"{index+1}, {pred.item()}\n")

def create_checkpoints_list(epochs_between_checkpoints, epochs):
    checkpoints_list = []
    epochs_executed = epochs_between_checkpoints
    while (epochs_executed <= epochs):
        checkpoints_list.append(epochs_executed)
        epochs_executed += epochs_between_checkpoints
    return checkpoints_list

def plot_fold_losses(train_losses: List[float], validation_losses: List[float], fold: int, folder: str) -> None:

    x = np.linspace(1, len(train_losses), len(train_losses))
    fig, axs = plt.subplots(2)
    
    axs[0].set_title("Training Loss")
    axs[0].plot(x, train_losses, c='c')

    axs[1].set_title("Validation Loss")
    axs[1].plot(x, validation_losses, c='m')
    
    plt.tight_layout()
    plt.savefig(folder+f"fold{fold}-training-test-loss.pdf")

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_device_number}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)

    get_model = None
    if args.model == "alexnet":
        get_model = get_alexNet
    if args.model == "imagenetalexnet":
        get_model = get_imagenet_alexNet
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
    elif args.model == "vggnet11":
        get_model = get_vggnet11
    elif args.model == "vggnet11pretrained":
        get_model = get_vggnet11_pretrained
    elif args.model == "MaCNN":
        get_model = get_MaCNN
    elif args.model == "lfcnn":
        get_model = get_lfCnn
    elif args.model == "resnext50":
        get_model = get_resnext50
    elif args.model == "resnext101":
        get_model = get_resnext101
    elif args.model == "resnext50pretrained":
        get_model = get_resnext50_pretrained
    elif args.model == "resnext101pretrained":
        get_model = get_resnext101_pretrained
    elif args.model == "darknet53":
        get_model = get_darknet53

    folder = args.experiment_folder

    make_dir(folder)

    n = len(EmbrapaP2Dataset(args.dataset_folder))

    folds = get_folds(n, 10)

    # Registro de informações de cada fold
    raw_fold_info = {
        "fold": [],
        "epoch": [],
        "train_loss": [],
        "validation_loss": []
    }

    # Métricas de cada fold
    fold_metrics_info = {
        "fold": [],
        "loss": [],
        "over": [],
        "under": [],
        "mean_error": [],
        "MAE": [],
        "MSE": [],
        "MAPE": [],
        "RMSE": [],
        "Pearson Correlation": []
    }

    # Informações de predição
    predictions_info = {
        "test_index": [],
        "prediction": [],
        "real_value": []
    }

    for k in range(10):

        mylogfile = folder + f"fold{k}.log"
        print_and_log((f"Fold #{k}",), mylogfile)

        # Preparação dos folds
        train_indexes = []
        for n in range(10):
            if n != k:
                train_indexes += folds[n]

        test_indexes = folds[k]

        # Criação dos datasets de treino e validação
        dstrain = EmbrapaP2Dataset(args.dataset_folder, train_indexes, augment=args.augment)
        dstest = EmbrapaP2Dataset(args.dataset_folder, test_indexes)

        # Criação dos DataLoaders de treino e avaliação
        dltrain = torch.utils.data.DataLoader(dstrain, shuffle=True, batch_size=args.batch_size)
        dltest = torch.utils.data.DataLoader(dstest, batch_size=args.batch_size)

        model = get_model()
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        chkpt_folder = f"{folder}fold{k}/"
        make_dir(chkpt_folder)

        checkpoints_list = create_checkpoints_list(args.epochs_between_checkpoints, args.epochs)

        training_start_time = time.time()

        # Treinamento em K-1 folds e avaliação no K-ésimo
        training_loss, test_loss = train(model, opt,  nn.MSELoss(), dltrain, dltest, 
                                         args.epochs, lr_schedular=None,
                                         cuda=True, logfile=mylogfile,
                                         checkpoints=checkpoints_list,
                                         checkpoints_folder=chkpt_folder)

        training_end_time = time.time()

        print_and_log((f"Training time: {training_end_time - training_start_time} seconds = {(training_end_time - training_start_time)/60} minutes", "\n"), mylogfile)

        evaluation_start_time = time.time()
        predictions, loss = evaluate(model, dltest, nn.MSELoss())
        real_values = [target for _, target in dstest]
        evaluation_end_time = time.time()
        
        print_and_log((f"Evaluation time: {evaluation_end_time - evaluation_start_time} seconds = {(evaluation_end_time - evaluation_start_time)/60} minutes", "\n"), mylogfile)

        print_and_log((f"Test Loss: {loss}", "\n"), mylogfile)

        plot_fold_losses(training_loss, test_loss, k, folder)

        # Atualizando informações a serializar
        raw_fold_info["fold"].extend([k]*len(training_loss))
        raw_fold_info["epoch"].extend(list(range(len(training_loss))))
        raw_fold_info["train_loss"].extend(training_loss)
        raw_fold_info["validation_loss"].extend(test_loss)

        metrics = get_metrics(real_values, predictions)
        fold_metrics_info["fold"].append(k)
        fold_metrics_info["loss"].append(loss)
        for metric_name, value in metrics.items():
            fold_metrics_info[metric_name].append(value)
        
        predictions_info["test_index"].extend([index + 1 for index in test_indexes])
        predictions_info["prediction"].extend(predictions)
        predictions_info["real_value"].extend(real_values)

        # Atualizando arquivos de serialização
        save_info(raw_fold_info, folder + "raw_fold_info.csv")
        save_info(fold_metrics_info, folder + "fold_metrics.csv")
        save_info(predictions_info, folder + "predictions.csv")
        
        if(args.only_one_fold and k == 0):
            sys.exit()