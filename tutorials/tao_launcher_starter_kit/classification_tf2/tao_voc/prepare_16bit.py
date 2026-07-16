 # Convert RGB images to (fake) 16-bit grayscale
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join as join_path


def to16bit(images_dir, img_file, images_dir_16_bit):
    image = Image.open(os.path.join(images_dir,img_file)).convert("L")
    # shifted to the higher byte to get a fake 16-bit image
    image_np = np.array(image) * 255
    image16 = Image.fromarray(image_np.astype(np.uint32))
    # overwrite the image file
    img_file = os.path.splitext(img_file)[0] + '.png'
    image16.save(os.path.join(images_dir_16_bit, img_file))


# Generate 16-bit grayscale images for train/val splits
DATA_DIR = os.environ.get('LOCAL_DATA_DIR')
os.makedirs(os.path.join(DATA_DIR, "training", "image_2_16bit_grayscale"), exist_ok=True)

source_dir = join_path(DATA_DIR, "VOCdevkit/VOC2012")
images_dir = join_path(source_dir, "JPEGImages")
images_dir_16_bit = images_dir.replace('JPEGImages','JPEGImages_16bit_grayscale')
os.makedirs(images_dir_16_bit, exist_ok=True)

for img_file in tqdm(os.listdir(images_dir)):
    to16bit(images_dir,img_file,images_dir_16_bit)

im = Image.open(join_path(images_dir_16_bit,'2008_007890.png'))
print("size:",im.size)
print("mode:",im.mode)
print("format:",im.format)
print(np.array(im).astype(np.uint32).shape)
