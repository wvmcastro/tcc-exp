import os
from argparse import ArgumentParser
from typing import Tuple, Dict
from math import sqrt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.lines as mlines

# Atualizar conforme a necessidade
COLOR_MAP = {
    "AlexNet": 'c', 
    "ResNet18":'r', 
    'VGGNet11': 'g',
    "MaCNN": "r",
    "ResNext50": "c",
    "ResNext101": "g"
}


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_metrics(real: Tuple, pred: Tuple) -> Tuple:
    unders = []
    overs = []
    for p in zip(pred, real):
        error = p[0] - p[1]
        if error > 0:
            overs.append(error)
        else:
            unders.append(error)
    
    n = len(real)

    # TODO: Usar métricas do sklearn...
    over = np.sum(overs)
    under = np.sum(unders)
    mean_error = (over + under) / n
    mean_abs_error = (over - under) / n
    mse = np.sum([e**2 for e in overs+unders])
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(real, pred)
    correlation = pearsonr(real, pred)[0]

    metrics = {"over": over, "under": under, "mean_error": mean_error,
               "MAE": mean_abs_error, "MSE": mse, "MAPE": mape, "RMSE": rmse, "Pearson Correlation": correlation}
    
    return metrics


def plot_rroc_space(metrics: dict, dstdir: str, aliases: Dict[str, str]):

    # TODO: Fazer o parsing do nome do modelo auqi dentro

    x = []
    y = []
    names = []
    for e, m in metrics.items():
        x.append(m["over"]/330)
        y.append(m["under"]/330)
        names.append(f"#{e}")
    
    l = max(np.max(x), abs(np.min(y)))

    plt.figure()
    plt.title("RROC SPACE")
    # under + over = 0
    dashes = [5, 5, 5, 5]
    p = np.linspace(0, 1.8*l, 100)
    plt.plot(p, -p, dashes=dashes, color="#cccccc")

    # colors = {"AlexNet": 'c', "ResNet18":'r', 'VGGNet11': 'g'}

    plt.xlim((0, 1.1*l))
    plt.ylim((-1.1*l, 0))
    plt.xlabel("OVER")
    plt.ylabel("UNDER")

    print(names)


    # plotting point
    for i, p in enumerate(zip(x,y)):
        # index = int(names[i].strip('#'))
        # model = experiment_model[index]
        print(names[i][1:], COLOR_MAP[names[i][1:]])
        plt.plot(p[0], p[1], COLOR_MAP[names[i][1:]]+'x', label=names[i] , markersize='12.0', markeredgewidth=2.0)
    
    # plotting name beside point
    for i, name in enumerate(names):
        # model = experiment_model[int(name.strip("#"))]
        # TODO: Refatorar para incluir os alias dos modelos !!!
        plt.text(x[i]+4, y[i]+4, "", color='k', fontsize=9)

    legend_elements = [
        mlines.Line2D([], [], color=color,  marker='x', linestyle='None', label=model, markersize='12.0', markeredgewidth=2.0) for model, color in COLOR_MAP.items()
    ]

    plt.legend(handles=legend_elements)

    plt.savefig(f"{dstdir}rroc.pdf")
    plt.close()


def get_avg_validation_loss(raw_fold_info_df: pd.DataFrame) -> pd.DataFrame:
    
    folds = len(raw_fold_info_df["fold"].unique())

    new_df = raw_fold_info_df[["epoch", "train_loss", "validation_loss"]].groupby("epoch").sum()
    new_df["train_loss"] /= folds
    new_df["validation_loss"] /= folds

    return new_df


def get_all_experiments_avg_validation_loss(experiment_folder: str, show_model_names: bool = True) -> None:

    experiments_df = pd.DataFrame(columns=["epoch", "train_loss", "validation_loss", "model"])

    for experiment in os.listdir(experiment_folder):
        for root, dirs, files in os.walk(experiment_folder+experiment):
            for file_name in files:

                if "raw_fold_info.csv" in file_name:

                    raw_info_df = pd.read_csv(os.path.join(experiment_folder, experiment, file_name))
                    experiment_avg_loss = get_avg_validation_loss(raw_info_df).reset_index()
                    experiment_avg_loss["model"] = experiment

                    experiments_df = experiments_df.append(experiment_avg_loss, ignore_index=True)

                    break

    models = experiments_df["model"].unique()
    models.sort()
    aliases = {name: f"#{i}" for i, name in enumerate(models, start=1)}

    if not show_model_names:
        experiments_df.replace(aliases, inplace=True)
    
    return experiments_df, aliases

def get_model_name_alias(model_name: str) -> str:

    for model_alias in COLOR_MAP.keys():
        if model_alias.lower() in model_name:
            return model_alias


def plot_and_save_rroc_curve(experiment_folder: str, aliases: Dict[str, str]):

    metrics = {}
    for experiment in os.listdir(experiment_folder):
        for root, dirs, files in os.walk(experiment_folder+experiment):
            for file_name in files:

                if "predictions.csv" in file_name:
                    # predictions_csv = pd.read_csv(os.path.join(experiment_folder, experiment, file_name))
                    # Gambito antes de teste
                    predictions_csv = pd.read_csv("/home/kenzo/experiments/alexnet-pretrained-50ep/predictions.csv")

                    # Sempre garantir que o nome das pastas dos experimentos contenham os nomes do COLOR_MAP
                    # experiment = get_model_name_alias(experiment)
                    metrics[experiment] = get_metrics(predictions_csv["real_value"].values, predictions_csv["prediction"].values)
    
                    break
    plot_rroc_space(metrics, experiment_folder, aliases)

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="directory with all experiment directories")
    parser.add_argument("--ommit_names", action="store_true", help="wheter ommit or not model names")
    parser.add_argument("--limited_epochs", type=int, help="specify number of epochs to be plotted, if not specified, plots all epochs.")
    args = parser.parse_args()

    experiments_df, aliases = get_all_experiments_avg_validation_loss(args.exp_dir, show_model_names=not args.ommit_names)
    experiments_df = experiments_df[["epoch", "validation_loss", "model"]]

    if args.limited_epochs is not None:
        experiments_df = experiments_df[experiments_df["epoch"] < args.limited_epochs] # filtrando epoca

    experiments_df.set_index("epoch", inplace=True)
    experiments_df = experiments_df.pivot(columns="model")
    
    experiments_df["validation_loss"].plot()
    plt.ylabel("Validation Loss")
    plt.xlabel("Epochs")

    # Saving plot and aliases in the experiment folder!
    plt.savefig(args.exp_dir + 'all-experiments-avg-val-loss.pdf')

    plot_and_save_rroc_curve(args.exp_dir, aliases)

    with open(args.exp_dir + "model_aliases.json", "w") as f:
        json.dump(aliases, f, indent=2)
