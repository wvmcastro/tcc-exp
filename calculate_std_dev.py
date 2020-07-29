import os
from argparse import ArgumentParser
import numpy as np
from typing import Tuple
from scipy.stats import pearsonr
from natsort import natsorted, ns
from math import sqrt
NUMBER_OF_FOLDS = 10

def make_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("predictions_srcdir", type=str)
    parser.add_argument("real_values_filename", type=str)
    parser.add_argument("--delimiter", type=str, default=',')
    return parser

def read_csv(raw_predictions_filename: str, real_values_filename: str, delimiter) -> Tuple[Tuple, Tuple]:
    real = []
    pred = []

    real_values_ordered = dict()
    with open(real_values_filename, 'r') as csv:
        for line in csv:
            values = line.split(delimiter)
            real_values_ordered[int(values[0])] = float(values[1].replace("\n", ""))

    with open(raw_predictions_filename, 'r') as csv:
        for line in csv:
            values = line.split(delimiter)
            real.append(float(real_values_ordered[int(values[0])]))
            pred.append(float(values[1]))
    
    return tuple(real), tuple(pred)

def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_std_devs(real: Tuple, pred: Tuple) -> Tuple:
    all_mean_errors = []
    all_mean_abs_errors = []
    all_mse = []
    all_mape = []
    all_correlations = []
    all_rmse = []
    real_divided_by_fold = split(real, NUMBER_OF_FOLDS)
    pred_divided_by_fold = split(pred, NUMBER_OF_FOLDS)

    for fold_index, real_values_in_fold in enumerate(real_divided_by_fold):
        n = len(real_values_in_fold)
        unders = []
        overs = []
        for p in zip(real_values_in_fold, pred_divided_by_fold[fold_index]):
            error = p[0] - p[1]
            if error > 0:
                overs.append(error)
            else:
                unders.append(error)
        
        over = np.sum(overs)
        under = np.sum(unders)
        mean_error = (over + under) / n
        all_mean_errors.append(mean_error)
        mean_abs_error = (over - under) / n
        all_mean_abs_errors.append(mean_abs_error)
        mse = np.sum([e**2 for e in overs+unders])
        all_mse.append(mse)
        rmse = srqt(mse)
        all_rmse.append(rmse)
        mape = mean_absolute_percentage_error(real_values_in_fold, pred_divided_by_fold[fold_index])
        all_mape.append(mape)
        correlation = pearsonr(real_values_in_fold, pred_divided_by_fold[fold_index])[0]
        all_correlations.append(correlation)

    std_devs = {"mean_error": np.std(all_mean_errors),
               "MAE": np.std(all_mean_abs_errors), "MSE": np.std(all_mse), "MAPE": np.std(all_mape), "RMSE": np.std(all_rmse), "Pearson Correlation": np.std(all_correlations)}
    
    return std_devs


if __name__ == "__main__":
    parser = make_parse()
    args = parser.parse_args()

    std_devs_by_metric = dict()
    for root, _, files in os.walk(args.predictions_srcdir):

        files = natsorted(files, alg=ns.IGNORECASE)
        for f in files:
            real, pred = read_csv(root+f, args.real_values_filename, args.delimiter)

            experiment_name = f.split('.')[0].replace('#', '')
            
            std_devs_by_metric[experiment_name] = get_std_devs(real, pred)
    
    print('>>>>>>> Standard Deviations by metric <<<<<<<')
    for key, values in std_devs_by_metric.items():
        # print(key)
        # print(values)
        print(f'{key.strip("experiment")} & {values["MAE"]} & {values["MSE"]} & {values["Pearson Correlation"]} & {values["MAPE"]} &{values["RMSE"]}')
        # print("-"*15)
        print()