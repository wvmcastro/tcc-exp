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
    parser.add_argument("--limited_epochs", type=int, help="specify number of epochs to be plotted, if not specified, plots all epochs.")
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
                
                if args.limited_epochs is not None:
                    number_of_epochs = args.limited_epochs
                else:
                    number_of_epochs = len(losses_object['fold#0']["validation_loss"])
                
                x = np.linspace(1, number_of_epochs, number_of_epochs)
                avg_val_losses_by_experiment[experiment_name] = np.zeros(number_of_epochs)

                for fold in range(NUMBER_OF_FOLDS):
                    avg_val_losses_by_experiment[experiment_name] += np.array(losses_object[f'fold#{fold}']["validation_loss"][:number_of_epochs])

                avg_val_losses_by_experiment[experiment_name] /= NUMBER_OF_FOLDS
                plt.plot(x, avg_val_losses_by_experiment[experiment_name], label=f'#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}')

    # ordering labels in legend by experiment number
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index('#1'), labels.index('#2'), labels.index('#3'), labels.index('#4'), labels.index('#5'), labels.index('#6'),
     labels.index('#7'), labels.index('#8'), labels.index('#9'), labels.index('#10'), labels.index('#11'), labels.index('#12')]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.savefig('all-experiments-avg-val-loss.pdf')