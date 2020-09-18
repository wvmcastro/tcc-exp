import os
from argparse import ArgumentParser
from my_utils import make_dir
from shutil import copyfile

# Usar estas strings abaixo como nome de projetos gerados pelo embrapa_experiment.py!

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
    'resnet18-pretrained-super-augmented': 12,
    'vggnet11-pretrained-super-augmented': 13,
    'vggnet11-super-augmented': 14,
    'MaCNN-super-augmented': 15,
    'lfcnn-super-augmented': 16,
    'resnext50': 17,
    'resnext50-augmented': 18,
    'resnext50-super-augmented': 19,
    'resnext50-pretrained': 20,
    'resnext50-pretrained-augmented': 21,
    'resnext50-pretrained-super-augmented': 22,
    'resnext101': 23,
    'resnext101-augmented': 24,
    'resnext101-super-augmented': 25,
    'resnext101-pretrained': 26,
    'resnext101-pretrained-augmented': 27,
    'resnext101-pretrained-super-augmented': 28
}

def build_ordered_csv_predictions_file_with_id_real_pred_columns(raw_predictions_filename: str, real_values_filename: str,
     csv_file_path):
    real_values_ordered = dict()
    pred_values_ordered = dict()
    delimiter = ','

    with open(real_values_filename, 'r') as csv:
        for line in csv:
            values = line.split(delimiter)
            real_values_ordered[int(values[0])] = float(values[1].replace("\n", ""))

    with open(raw_predictions_filename, 'r') as csv:
        for line in csv:
            values = line.split(delimiter)
            pred_values_ordered[int(values[0])] = float(values[1].replace("\n", ""))
            

    csv_to_write = open(csv_file_path, "w+")
    for i in range(1, len(real_values_ordered) + 1):
        csv_to_write.write(f"{i}, {real_values_ordered[i]}, {pred_values_ordered[i]}\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src_dir", type=str, help="directory with all experiment directories")
    parser.add_argument("out_dir", type=str, help="directory to be created to store csvs")
    parser.add_argument("--real_values_filename", type=str, default="", help="csv file with real values")
    args = parser.parse_args()
    raw_named_csvs_dir = f'{args.out_dir}/raw-named-csvs'
    raw_numbered_csvs = f'{args.out_dir}/raw-numbered-csvs'
    make_dir(args.out_dir)
    make_dir(raw_named_csvs_dir)
    make_dir(raw_numbered_csvs)
    if args.real_values_filename != "":
        id_real_pred_numbered_csvs = f'{args.out_dir}/id-real-pred-numbered-csvs'
        make_dir(id_real_pred_numbered_csvs)

    for root, dirs, files in os.walk(args.src_dir):
        for dir in dirs:
            csv_file = f'{root}/{dir}/predictions.csv'
            experiment_name = dir
            copyfile(csv_file, f'{raw_named_csvs_dir}/{experiment_name}.csv')
            copyfile(csv_file, f'{raw_numbered_csvs}/experiment#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}.csv')

            if args.real_values_filename != "":
                build_ordered_csv_predictions_file_with_id_real_pred_columns(csv_file, args.real_values_filename,
                 f'{id_real_pred_numbered_csvs}/experiment#{EXPERIMENT_NUMBER_BY_NAME[experiment_name]}.csv')



            




