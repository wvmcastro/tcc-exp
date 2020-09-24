import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

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

    with open(args.exp_dir + "model_aliases.json", "w") as f:
        json.dump(aliases, f, indent=2)