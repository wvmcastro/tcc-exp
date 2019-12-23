import os
from typing import Tuple
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   
import numpy as np
from my_utils import make_dir

def make_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("srcdir", type=str)
    parser.add_argument("dstdir", type=str)
    parser.add_argument("--delimiter", type=str, default=';')
    parser.add_argument("--bins", type=int, default=20)
    return parser

def read_csv(filename: str, delimiter) -> Tuple[Tuple, Tuple]:
    real = []
    pred = []

    with open(filename, 'r') as csv:
        for line in csv:
            values = line.split(delimiter)
            real.append(float(values[1]))
            pred.append(float(values[2]))
    
    return tuple(real), tuple(pred)

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
    n1, bins, _ = plt.hist(real, bins=args.bins,
                                       weights=w, 
                                       facecolor="#34a2eb",
                                       edgecolor="#2c5aa3",
                                       alpha=0.9)
    
    n2, bins, _ = plt.hist(pred, bins=args.bins, 
                                       weights=w,
                                       facecolor="#ffbc47",
                                    #    edgecolor="#9e742b", 
                                       alpha=0.6)
    
    real_patch = mpatches.Patch(color='#34a2eb', label='y')
    pred_patch = mpatches.Patch(color='#ffbc47', label='ŷ')
    plt.legend(handles=[real_patch, pred_patch])
    
    intersection = calc_intersections(n1, n2)
    plot_nane = f"{experiment_name}-hist-{intersection}.pdf"
    plt.savefig(plot_nane, bbox_inches="tight")

def scatter_plot_and_save(experiment_name: str,
                          real: Tuple,
                          pred: Tuple) -> None:
    
    plt.figure()
    plt.xlabel("REAL")
    plt.ylabel("PREDITO")

    plt.plot(real, pred, 'co')
    
    dashes = [5, 5, 5, 5]
    plt.plot(real, real, dashes=dashes, color="#cccccc")

    plot_name = f"{experiment_name}-scatter.pdf"
    plt.savefig(plot_name, bbox_inches="tight")

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

    over = np.sum(overs)
    under = np.sum(unders)
    mean_error = (over + under) / n
    mean_abs_error = (over - under) / n
    mse = np.sum([e**2 for e in overs+unders])

    metrics = {"over": over, "under": under, "mean_error": mean_error,
               "MAE": mean_abs_error, "MSE": mse}
    
    return metrics


def plot_rroc_space(metrics: dict, dstdir):
    x = []
    y = []
    names = []
    for e, m in metrics.items():
        x.append(m["over"]/330)
        y.append(m["under"]/330)
        names.append(f"#{e[-1]}")
    
    l = max(np.max(x), abs(np.min(y)))

    plt.figure()
    plt.title("ESPAÇO RROC")
    # under + over = 0
    dashes = [5, 5, 5, 5]
    p = np.linspace(0, 1.8*l, 100)
    plt.plot(p, -p, dashes=dashes, color="#cccccc")

    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    plt.xlim((0, 1.1*l))
    plt.ylim((-1.1*l, 0))
    plt.xlabel("SUPERIOR")
    plt.ylabel("INFERIOR")
    
    for i, p in enumerate(zip(x,y)):
        plt.plot(p[0], p[1], colors[i]+'x')
    
    for i, name in enumerate(names):
        plt.text(x[i]+4, y[i]+4, name, color=colors[i], fontsize=9)

    plt.savefig(f"{dstdir}rroc.pdf")



if __name__ == "__main__":
    parser = make_parse()
    args = parser.parse_args()

    make_dir(args.dstdir)

    metrics = dict()
    w = None
    for root, _, files in os.walk(args.srcdir):

        for f in files:
            real, pred = read_csv(root+f, args.delimiter)

            if w is None:
                w = np.ones(len(real)) / len(real)

            experiment_name = f.split('.')[0].replace('#', '')
            experiment_file = args.dstdir + experiment_name
            
            metrics[experiment_name] = get_metrics(real, pred)

            # plot_and_save_histogram(experiment_file, 
            #                         real, pred, 
            #                         args.bins, w)
            # scatter_plot_and_save(experiment_file, real, pred)
    
    for key, values in metrics.items():
        print(key)
        print(values)
        print("-"*15)
        print()
    
    plot_rroc_space(metrics, args.dstdir)

