import os
import shutil
from tqdm import tqdm

DATA_DIR=os.environ.get('LOCAL_DATA_DIR')

with open("imagenet_valprep.txt", "r") as f:
    for line in tqdm(f):
        img_name, dir_name = line.rstrip().split(" ")
        target_dir = os.path.join(DATA_DIR, "imagenet", "val", dir_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(os.path.join(DATA_DIR, "imagenet", "val", img_name), os.path.join(target_dir, img_name))

# This results in a validation directory like so:
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
