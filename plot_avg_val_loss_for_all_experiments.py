import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

NUMBER_OF_FOLDS = 10
EXPERIMENT_NUMBER_BY_NAME = {
    'alexnet': 1,
    'alexnet-augmented': 2,
    'alexnet-super-augmented': 3,
    'resnet18': 4,
    'resnet18-augmented': 5,
    'resnet18-super-augmented': 6,
    'myalexnet-pretrained': 7,
    'myalexnet-pretrained-augmented': 8,
    'myalexnet-pretrained-super-augmented': 9,
    'resnet18-pretrained': 10,
    'resnet18-pretrained-augmented': 11,
    'resnet18-pretrained-super-augmented': 12
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="directory with all experiment directories")
    args = parser.parse_args()

    avg_val_losses_by_experiment = dict()
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")

    for root, dirs, files in os.walk(args.exp_dir):
        for dir in dirs:
            losses_file = f'{root}/{dir}/losses.bin'
            experiment_name = dir

            with open(losses_file, "rb") as fp:
                losses_object = pk.load(fp)
                number_of_epochs = len(losses_object['fold#0'])
                x = np.linspace(1, number_of_epochs, number_of_epochs)
                avg_val_losses_by_experiment[experiment_name] = np.zeros(number_of_epochs)

                for fold in range(NUMBER_OF_FOLDS):
                    avg_val_losses_by_experiment[experiment_name] += np.array(losses_object[f'fold#{fold}']["validation_loss"])

                avg_val_losses_by_experiment[experiment_name] /= number_of_epochs
                plt.plot(x, avg_val_losses_by_experiment[experiment_name], label=f'#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}')

    plt.legend()
    plt.savefig('all-experiments-avg-val-loss.pdf')