from argparse import ArgumentParser
from datasets import EmbrapaP2Dataset

import torch    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder", type=str)
    
    args = parser.parse_args()

    dataset = EmbrapaP2Dataset(args.folder, 227)
    print(len(dataset))

    dl = torch.utils.data.DataLoader(dataset)

    for x, y in dl:
        print(x.shape, y)