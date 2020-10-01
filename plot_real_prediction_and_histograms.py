import os
from typing import Tuple
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import pearsonr
from math import sqrt
import pandas as pd

from my_utils import make_dir

def make_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("dst_dir", type=str)
    parser.add_argument("--bins", type=int, default=20)
    return parser


def calc_intersections(hist1, hist2) -> float:
    s = 0
    for p in zip(hist1, hist2):
        s += min(p)
    
    return s

def plot_and_save_histogram(experiment_name: str,
                            real: Tuple, 
                            pred: Tuple, 
                            bins: int, 
                            weights=None) -> None:
    plt.figure()
    
    # controls default text sizes
    plt.rc('font', size=16)          
    
    range_min = min(np.min(real), np.min(pred))
    range_max = max(np.max(real), np.max(pred))

    # hard-coding axis' max values (min, max)
    # plt.ylim(0, 0.3)

    full_range = (range_min, range_max)

    n1, bins, _ = plt.hist(real, bins=bins,
                                 range=full_range,
                                 weights=weights, 
                                 facecolor="#34a2eb",
                                 edgecolor="#2c5aa3",
                                 alpha=0.9)
    
    n2, bins, _ = plt.hist(pred, bins=bins, 
                                 range=full_range,
                                 weights=weights,
                                 facecolor="#ffbc47",
                              #    edgecolor="#9e742b", 
                                 alpha=0.6)
    
    real_patch = mpatches.Patch(color='#34a2eb', label='y')
    pred_patch = mpatches.Patch(color='#ffbc47', label='ŷ')
    plt.legend(handles=[real_patch, pred_patch])
    
    intersection = calc_intersections(n1, n2)
    plot_nane = f"{experiment_name}-hist-{intersection}.pdf"
    plt.savefig(plot_nane, bbox_inches="tight")
    plt.close()


def scatter_plot_and_save(experiment_name: str,
                          real: Tuple,
                          pred: Tuple) -> None:
    
    plt.figure()

    # controls default text sizes
    plt.rc('font', size=16)          

    # hard-coding axis' max values [xmin, xmax, ymin, ymax]
    # plt.axis([0, 15000, 0, 17500])

    plt.xlabel("REAL")
    plt.ylabel("PREDICTION")

    plt.plot(real, pred, 'co')    
    dashes = [5, 5, 5, 5]
    
    plt.plot(real, real, dashes=dashes, color="#cccccc")

    plot_name = f"{experiment_name}-scatter.pdf"
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close()

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


def plot_rroc_space(metrics: dict, dstdir):

    # TODO: Refatorar essa belezinha para funcionar com um exemplo só

    x = []
    y = []
    names = []
    for e, m in metrics.items():
        x.append(m["over"]/330)
        y.append(m["under"]/330)
        names.append(f"#{e.strip('experiment')}")
    
    l = max(np.max(x), abs(np.min(y)))

    plt.figure()
    plt.title("RROC SPACE")
    # under + over = 0
    dashes = [5, 5, 5, 5]
    p = np.linspace(0, 1.8*l, 100)
    plt.plot(p, -p, dashes=dashes, color="#cccccc")

    colors = {"AlexNet": 'c', "ResNet18":'r', 'VGGNet11': 'g'}

    plt.xlim((0, 1.1*l))
    plt.ylim((-1.1*l, 0))
    plt.xlabel("OVER")
    plt.ylabel("UNDER")

    # plotting point
    for i, p in enumerate(zip(x,y)):
        index = int(names[i].strip('#'))
        model = experiment_model[index]
        plt.plot(p[0], p[1], colors[model]+'x', label=names[i] + " " + model, markersize='12.0', markeredgewidth=2.0)
    
    # plotting name beside point
    for i, name in enumerate(names):
        model = experiment_model[int(name.strip("#"))]
        plt.text(x[i]+4, y[i]+4, name, color='k', fontsize=9)

    legend_elements = [mlines.Line2D([], [], color='c',  marker='x', linestyle='None', label='AlexNet',markersize='12.0', markeredgewidth=2.0),
     mlines.Line2D([], [], color='r',  marker='x', linestyle='None', label='ResNet18', markersize='12.0', markeredgewidth=2.0),
     mlines.Line2D([], [], color='g',  marker='x', linestyle='None', label='VGGNet11', markersize='12.0', markeredgewidth=2.0)]

    plt.legend(handles=legend_elements)

    plt.savefig(f"{dstdir}rroc.pdf")
    plt.close()


if __name__ == "__main__":
    parser = make_parse()
    args = parser.parse_args()

    make_dir(args.dst_dir)

    predictions_df = pd.read_csv(args.exp_dir + "predictions.csv")

    experiment_name = args.exp_dir.split("/")[-2]
    experiment_file = args.dst_dir + "/" + experiment_name

    real, pred = predictions_df["real_value"].values, predictions_df["prediction"].values
    
    w = np.ones(len(real)) / len(real)

    # Histogram plot
    plot_and_save_histogram(experiment_file, 
                            real, pred, 
                            args.bins, w)

    # Real x Prediction plot
    scatter_plot_and_save(experiment_file, real, pred)
