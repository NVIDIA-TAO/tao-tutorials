# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Script to visualize the Ground truth masks overlay for Unet tutorial."""

import os
import random
import argparse
import cv2
import numpy as np

def get_color_id(num_classes):
    """Function to return a list of color values for each class."""

    colors = []
    for idx in range(num_classes):
        random.seed(idx)
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors


def overlay_seg_image(inp_img, seg_img):
    """The utility function to overlay mask on original image."""

    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    overlayed_img = (inp_img/2 + seg_img/2).astype('uint8')
    return overlayed_img


def visualize_masks(img_dir, mask_dir, vis_dir, num_imgs=None, num_classes=2):
    """The function to visualize the segmentation masks.

    Args:
        img_dir: Directory of images.
        mask_dir: Mask images annotations.
        vis_dir: The output dir to save visualized images.
        num_classes: number of classes.
        num_imgs: number of images to visualize.
    """

    # Create the visualization dir
    if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)

    colors = get_color_id(num_classes)
    img_names = os.listdir(img_dir)
    if not num_imgs:
        num_imgs = len(img_names)
    mask_sample_name = os.listdir(mask_dir)[0]
    mask_ext = mask_sample_name.split(".")[-1]

    for img_name in img_names[:num_imgs]:
        img_path = os.path.join(img_dir, img_name)
        orig_image = cv2.imread(img_path)
        output_height = orig_image.shape[0]
        output_width = orig_image.shape[1]
        segmented_img = np.zeros((output_height, output_width, 3))
        pred = cv2.imread(os.path.join(mask_dir, img_name.split(".")[0]+"."+mask_ext),0)
        for c in range(len(colors)):
            seg_arr_c = pred[:, :] == c
            segmented_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
            segmented_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
            segmented_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
        fused_img = overlay_seg_image(orig_image, segmented_img)
        cv2.imwrite(os.path.join(vis_dir, img_name), fused_img)


def build_command_line_parser():
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Visualize Segmentation.', description='Overlay Segmentation.')

    parser.add_argument(
        '-i',
        '--imgs_dir',
        type=str,
        default=None,
        help='Path to folder where images are saved.'
    )
    parser.add_argument(
        '-m',
        '--masks_dir',
        type=str,
        default=None,
        help='Path to a folder where mask images are saved.'
    )
    parser.add_argument(
        '-o',
        '--vis_dir',
        type=str,
        default=None,
        help='Path to a folder where the segmentation overlayed images are saved.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Number of classes.'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help='Number of images to visualize.'
    )

    return parser


def parse_command_line_args():
    """Parser command line arguments to the trainer.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser()
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line_args()
    visualize_masks(args.imgs_dir, args.masks_dir, args.vis_dir, args.num_images, args.num_classes)

if __name__ == '__main__':
    main()
