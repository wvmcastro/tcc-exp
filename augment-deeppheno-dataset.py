from typing import Tuple
from argparse import ArgumentParser
import os
from PIL import Image
from my_utils import make_dir

def augmentData(src: str, dst: str) -> None:
    for root, _, files in os.walk(src):
        if files != []:
            folder = root.split('/')[-1]
            path = dst + folder
            make_dir(path)

            for f in files:
                _00, _90, _180, _270 = augment_image(root+'/'+f)
                save_img(_00, path+'/'+f, "00")
                save_img(_90, path+'/'+f, "90")
                save_img(_180, path+'/'+f, "180")
                save_img(_270, path+'/'+f, "270")


def augment_image(img_path: str) -> Tuple:
    img = Image.open(img_path)
    _90 = img.rotate(90)
    _180 = img.rotate(180)
    _270 = img.rotate(270)

    return img, _90, _180, _270

def save_img(img, path: str, deg: str):
    name = path[:-4] + '_' + deg + ".jpg"
    img.save(name)

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    args = parser.parse_args()

    make_dir(args.dst)
    
    augmentData(args.src, args.dst)
    
