from embrapa_experiment import *
from argparse import ArgumentParser
import json 
from typing import List


def get_val_ids(val_ids_path: str) -> List[int]:

    with open(val_ids_path, "r") as f:
        val_ids = json.load(f)

    return val_ids

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("model", type=str, 
                        help="supported models: alexnet, resnet18 or vggnet11")
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("--valid_ids", type=str, required=True,
                        help="Path to a json file containing the parcel ID's to be used in the validation.")
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--augment", type=str, default="no", help="options: no, yes, super")
    parser.add_argument("--epochs_between_checkpoints", type=int, default=100)
    parser.add_argument("--cuda_device_number", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_device_number}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)

    get_model = get_model_factory(args.model)
    valid_ids = get_val_ids(args.valid_ids)
    
    folder = args.experiment_folder

    make_dir(folder)

    n = len(EmbrapaP2Dataset(args.dataset_folder))

    all_ids = list(range(n))
    train_ids = list(set(all_ids).difference(valid_ids))

    mylogfile = folder + f"holdout_logs.log"
    print_and_log((f"-- {args.model} experiment",), mylogfile)

    dstrain = EmbrapaP2Dataset(args.dataset_folder, train_ids, augment=args.augment)
    dstest = EmbrapaP2Dataset(args.dataset_folder, valid_ids)


    # Criação dos DataLoaders de treino e avaliação
    dltrain = torch.utils.data.DataLoader(dstrain, shuffle=True, batch_size=args.batch_size)
    dltest = torch.utils.data.DataLoader(dstest, batch_size=args.batch_size)

    model = get_model()
        
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    chkpt_folder = f"{folder}checkpoint/"
    make_dir(chkpt_folder)

    checkpoints_list = create_checkpoints_list(args.epochs_between_checkpoints, args.epochs)

    training_start_time = time.time()

    # Treinamento em K-1 folds e avaliação no K-ésimo
    training_loss, test_loss = train(model, opt,  nn.MSELoss(), dltrain, dltest, 
                                        args.epochs, lr_schedular=None,
                                        cuda=True, logfile=mylogfile,
                                        checkpoints=checkpoints_list,
                                        checkpoints_folder=chkpt_folder)

    training_end_time = time.time()

    print_and_log((f"Training time: {training_end_time - training_start_time} seconds = {(training_end_time - training_start_time)/60} minutes", "\n"), mylogfile)

    evaluation_start_time = time.time()
    predictions, loss = evaluate(model, dltest, nn.MSELoss())
    real_values = [target for _, target in dstest]
    evaluation_end_time = time.time()

    print_and_log((f"Evaluation time: {evaluation_end_time - evaluation_start_time} seconds = {(evaluation_end_time - evaluation_start_time)/60} minutes", "\n"), mylogfile)

    metrics = get_metrics(real_values, predictions)
    json.dump(metrics, open(folder + "/metrics.json", "w"), indent=2)

