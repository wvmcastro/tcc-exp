import sys
sys.path.append('..')

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import numpy as np
import pandas as pd

from typing import Tuple
from argparse import ArgumentParser
from my_utils import make_dir
from my_utils_regression import get_metrics

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


if __name__ == "__main__":
    parser = make_parse()
    args = parser.parse_args()

    make_dir(args.dst_dir)
    
    print(f"Loading from: {args.exp_dir}" + "predictions.csv")

    predictions_df = pd.read_csv(args.exp_dir + "predictions.csv")

    experiment_name = args.exp_dir.split("/")[-2]
    experiment_file = args.dst_dir + "/" + experiment_name

    real, pred = predictions_df["real_value"].values, predictions_df["prediction"].values

    print(get_metrics(predictions_df["real_value"].values, predictions_df["prediction"].values))
    
    w = np.ones(len(real)) / len(real)

    # Histogram plot
    plot_and_save_histogram(experiment_file, 
                            real, pred, 
                            args.bins, w)

    # Real x Prediction plot
    scatter_plot_and_save(experiment_file, real, pred)

