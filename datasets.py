import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import json
from collections import OrderedDict

class DeepPhenoDataset(data.Dataset):
    classes = {"Ler-1": 0, "Col-0": 1, "Sf-2": 2, "cvi": 3}
    def __init__(self, path: str, transform = None):
        super().__init__()
        self._list_IDS = None
        self._labels = None
        self._transforms = transform

        self._list_IDS, self._labels = self.get_data(path)
        self._pilToTensor = transforms.ToTensor()
    
    def walk_directory(self, path: str):
        data = {c: dict() for c in DeepPhenoDataset.classes.keys()}

        for root, _, files in os.walk(path):
            if files != []:
                for c in DeepPhenoDataset.classes:
                    if c in root:
                        d = data[c]
                        break
                
                l = []
                for f in files:
                    l.append(f)
                d[root] = sorted(l)

        return data

    def get_data(self, datapath: str) -> tuple:
        dataset = self.walk_directory(datapath)

        ids = []
        labels = []
        
        for c in dataset.keys():
            for partition in dataset[c].items():
                folder, files = partition
                for f in files:
                    ids.append(folder+"/"+f)
                    labels.append(DeepPhenoDataset.classes[c])
        
        return ids, labels
    
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, index) -> tuple:
        ID = self._list_IDS[index]
        
        x = Image.open(ID)
        if self._transforms is not None:
            x = self._transforms(x)
        else:
            x = self._pilToTensor(x)

        y = self._labels[index]

        return x, y

class EmbrapaP2Dataset(data.Dataset):
    def __init__(self, dataset_folder: str, 
                       indexes = None, 
                       augment: str = "no"):
        self._folder = dataset_folder
        self._list_IDS = []
        
        self._ys = []
        self._ymean = 0
        self._ystd = 0
        self._ymin = 0
        self._ymax = 0
        
        self._augment = augment

        self._transform = None
        self._load_dataset(dataset_folder, indexes)
    
    def _load_dataset(self, dataset_folder: str, indexes) -> None:
        annotation_file = "labels/annotation.json"
        path = dataset_folder + annotation_file

        with open(path, 'r') as fp:
            annotations = json.load(fp, object_pairs_hook=OrderedDict)

        # ystats = annotations["statistics"]["y"]
        # self._ymean, self._ystd = ystats["mean"], ystats["std"]
        # self._ymin, self._ymax = ystats["min"], ystats["max"]
        
        # s = self._ymax - self._ymin
        if indexes is None:
            for ID, y in annotations["data"].items():
                self._list_IDS.append(ID)
                self._ys.append(y)
        else:
            ann = list(annotations["data"].items())
            for index in indexes:
                ID, y = ann[index]
                self._list_IDS.append(ID)
                self._ys.append(y)
        
        xstats = annotations["statistics"]["x"]
        normalize = transforms.Normalize(mean=xstats["mean"], std=xstats["std"])

        t = transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
            normalize])
        
        self._transform = t

    def __len__(self):
        if self._augment == "super":
            return 3 * len(self._list_IDS)
        elif self._augment == "yes":
            return 2 * len(self._list_IDS)
        else:
            return len(self._list_IDS)

    def __getitem__(self, index) -> tuple:
        if self._augment == "super":
            true_index = index // 3
            y = self._ys[true_index]

            ID = self._list_IDS[true_index]
            x = Image.open(self._folder + ID)
	
            if index % 3 == 1:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
            elif index % 3 == 2:
                x = x.transpose(Image.FLIP_TOP_BOTTOM)
        elif self._augment == "yes":
            true_index = index // 2
            y = self._ys[true_index]

            ID = self._list_IDS[true_index]
            x = Image.open(self._folder + ID)

            if index % 2 == 1:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            ID = self._list_IDS[index]
            x = Image.open(self._folder + ID)
            y = self._ys[index]

        x = self._transform(x)
        

        return x, np.float32(y)

    # Cyclical learning hate com SGD
