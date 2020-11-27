from typing import List
from torch.utils.data.dataloader import DataLoader
from embrapa_experiment import *
from datasets import InferenceDataset
from argparse import ArgumentParser
import torch
from torchvision import transforms
import json
import os
import pandas as pd

# python3 predict.py --model myalexnetpretrained --checkpoint_path ../../experiments/alexnet-pretrained-super_aug-200ep/fold9/MyAlexNetPretrained_200.pth --annotations ../../../biomassa/exp/GSD05/2019-01-23/labels/annotation.json --image_folder ../../datasets/embrapa/Fake-P2-dataset/ --batch_size 32 --cuda_device_number 2

def get_transforms(xstats: dict):

    normalize = transforms.Normalize(mean=xstats["mean"], std=xstats["std"])

    t = transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
            normalize])
            
    return t


def predict(model, inference_dl: DataLoader, device: torch.device = "cpu") -> List[float]:
    predictions = []
    with torch.no_grad():
        for image_batch in inference_dl:

            image_batch = image_batch.to(device)
            preds = model(image_batch).view(-1)
            predictions.extend(preds.tolist())

    return predictions

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, 
                        help="supported models: alexnet, resnet18 or vggnet11")
    parser.add_argument("--checkpoint_path", type=str, 
                        help="Path to the model checkpoint.")
    parser.add_argument("--annotations", type=str,
                        help="Path to the file containing the image stats.")
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cuda_device_number", type=int, default=0)

    args = parser.parse_args()

    get_model = get_model_factory(args.model)

    model = get_model()
    device = torch.device(f'cuda:{args.cuda_device_number}') if torch.cuda.is_available() else torch.device('cpu')
    annotations = json.load(open(args.annotations, "r"))
    xstats = annotations["statistics"]["x"]
    transforms = get_transforms(xstats)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)["state_dict"], strict=False)

    ds = InferenceDataset(args.image_folder, transforms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    preds = predict(model.to(device), dl, device)

    preds_df = pd.DataFrame()
    # preds_df["images"] = ds.images
    preds_df["Predicao"] = preds
    preds_df["Imagem"] = ds.images

    preds_df.to_csv(os.path.join(args.image_folder, "predictions.csv"))
