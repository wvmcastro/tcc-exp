from typing import Tuple, List, Iterable, Dict
import os
from random import shuffle
import pandas as pd

from torch.utils import data

def make_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def get_folds(dataset_len: int, k: int) -> List:
    indexes = [i for i in range(dataset_len)]
    shuffle(indexes)

    step = dataset_len // k
    not_mutiple = k*step != dataset_len


    slices = []
    for z, i in enumerate(range(0, dataset_len, step)):
        if z == k-1 and not_mutiple:
            right = dataset_len
        else:
            right = i + step
        s = slice(i, right, 1)
        slices.append(indexes[s])

        if z == k-1:
            break

    return slices

def print_and_log(strings: Iterable, logfile = None) -> None:
    if logfile is not None:
        a = type(logfile) == str 
        if a == True:
            logfile = open(logfile, "a+")
        for s in strings:
            print(s)
            logfile.write(s)
            logfile.write("\n")
        if a == True:
            logfile.close()
    else:
        for s in strings:
            print(s)

def save_info(info: Dict, file_name: str) -> None:

    info_df = pd.DataFrame.from_dict(info)
    info_df.to_csv(file_name, index=False)