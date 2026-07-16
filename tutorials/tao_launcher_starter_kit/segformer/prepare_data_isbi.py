# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Script to prepare train/val dataset for Unet tutorial."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageSequence


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(description='Prepare train/val dataset for UNet tutorial')

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory to ISBI Tiff Files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Ouput directory to processes images from ISBI Tiff files."
    )

    return parser.parse_args(args)

def save_arrays_as_images(arr, im_dir):
    """Utility function to save the images to dir from arrays."""
    for idx, arr in enumerate(arr):
        img_name = os.path.join(im_dir, "image_{}.png".format(idx))
        cv2.imwrite(img_name, arr)

def load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

def check_and_create(d):
    """Utility function to create a dir if not present"""
    if not os.path.isdir(d):
        os.makedirs(d)

def main(args=None):
    """Main function for data preparation."""

    args = parse_args(args)
    
    print("Preparing Dataset Succesfully.")

    train_images_tif = os.path.join(args.input_dir, "train-volume.tif")
    train_masks_tif = os.path.join(args.input_dir, "train-labels.tif")
    test_images_tif = os.path.join(args.input_dir, "test-volume.tif")

    output_images_dir = os.path.join(args.output_dir,"images")
    output_masks_dir = os.path.join(args.output_dir,"masks")

    # Creating the images dir for train, test, val
    train_images_dir = os.path.join(output_images_dir,"train")
    val_images_dir = os.path.join(output_images_dir,"val")
    test_images_dir = os.path.join(output_images_dir,"test")

    train_masks_dir = os.path.join(output_masks_dir,"train")
    val_masks_dir = os.path.join(output_masks_dir,"val")

    check_and_create(train_images_dir)
    check_and_create(val_images_dir)
    check_and_create(test_images_dir)
    check_and_create(train_masks_dir)
    check_and_create(val_masks_dir)

    train_np_arrays_images = load_multipage_tiff(train_images_tif)
    train_np_arrays_masks = load_multipage_tiff(train_masks_tif)
    test_np_arrays_images = load_multipage_tiff(test_images_tif)

    # Splitting the train numpy arrays into train and val
    train_np_arrays_images_final = train_np_arrays_images[:20,:,:]
    train_np_arrays_masks_final = train_np_arrays_masks[:20,:,:]

    val_np_arrays_images_final = train_np_arrays_images[20:,:,:]
    val_np_arrays_masks_final = train_np_arrays_masks[20:,:,:]

    # Saving the train arrays as images
    save_arrays_as_images(train_np_arrays_images_final, train_images_dir)
    save_arrays_as_images(train_np_arrays_masks_final, train_masks_dir)

    # Saving the val arrays as images
    save_arrays_as_images(val_np_arrays_images_final, val_images_dir)
    save_arrays_as_images(val_np_arrays_masks_final, val_masks_dir)

    # Saving the test arrays as images
    save_arrays_as_images(test_np_arrays_images, test_images_dir)

    print("Prepared data successfully !")

if __name__ == "__main__":
    main()