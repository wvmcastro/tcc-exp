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

LINE_COLOR_BY_EXP = {
    'alexnet': '#a6cee3',
    'alexnet-augmented': '#1f78b4',
    'alexnet-super-augmented': '#b2df8a',
    'resnet18': '#33a02c',
    'resnet18-augmented': '#fb9a99',
    'resnet18-super-augmented': '#e31a1c',
    'myalexnet-pretrained': '#fdbf6f',
    'myalexnet-pretrained-augmented': '#ff7f00',
    'myalexnet-pretrained-super-augmented': '#cab2d6',
    'resnet18-pretrained': '#6a3d9a',
    'resnet18-pretrained-augmented': '#ffff99',
    'resnet18-pretrained-super-augmented': '#b15928'
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
                    epochs_start = 5
                    number_of_epochs = args.limited_epochs
                else:
                    epochs_start = 5
                    number_of_epochs = len(losses_object['fold#0']["validation_loss"])
                
                x = np.linspace(epochs_start, number_of_epochs, number_of_epochs - epochs_start)
                avg_val_losses_by_experiment[experiment_name] = np.zeros(number_of_epochs - epochs_start)


                for fold in range(NUMBER_OF_FOLDS):
                    avg_val_losses_by_experiment[experiment_name] += np.array(losses_object[f'fold#{fold}']["validation_loss"][epochs_start:number_of_epochs])

                avg_val_losses_by_experiment[experiment_name] /= NUMBER_OF_FOLDS

                # if (EXPERIMENT_NUMBER_BY_NAME[experiment_name] > 6):
                #     plt.plot(x, avg_val_losses_by_experiment[experiment_name], label=f'#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}', linestyle='dotted')
                # else:

                plt.plot(x, avg_val_losses_by_experiment[experiment_name], label=f'#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}', color=LINE_COLOR_BY_EXP[experiment_name])

    # ordering labels in legend by experiment number
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index('#1'), labels.index('#2'), labels.index('#3'), labels.index('#4'), labels.index('#5'), labels.index('#6'),
     labels.index('#7'), labels.index('#8'), labels.index('#9'), labels.index('#10'), labels.index('#11'), labels.index('#12')]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.savefig('all-experiments-avg-val-loss.pdf')