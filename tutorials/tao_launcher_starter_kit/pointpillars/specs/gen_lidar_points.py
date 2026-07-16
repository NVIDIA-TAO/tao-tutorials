# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

import os
import argparse

import numpy as np
from skimage import io

from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils.calibration_kitti import (
    Calibration
)


def parse_args():
    parser = argparse.ArgumentParser("Limit LIDAR points to FOV range.")
    parser.add_argument(
        "-p", "--points_dir",
        type=str, required=True,
        help="LIDAR points directory."
    )
    parser.add_argument(
        "-c", "--calib_dir",
        type=str, required=True,
        help="Calibration file directory"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str, required=True,
        help="Output LiDAR points directory"
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str, required=True,
        help="image directory"
    )
    return parser.parse_args()


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag


def generate_lidar_points(points_dir, calib_dir, output_dir, image_dir):
    """Limit LiDAR points to FOV range."""
    for pts in os.listdir(points_dir):
        pts_file = os.path.join(points_dir, pts)
        points = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
        calib_file = os.path.join(calib_dir, pts[:-4]+".txt")
        calib = Calibration(calib_file)
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        img_file = os.path.join(image_dir, pts[:-4]+".png")
        img_shape = np.array(io.imread(img_file).shape[:2], dtype=np.int32)
        fov_flag = get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]
        points.tofile(os.path.join(output_dir, pts))
        # double check
        points_cp = np.fromfile(os.path.join(output_dir, pts), dtype=np.float32).reshape(-1, 4)
        assert np.equal(points, points_cp).all()


if __name__ == "__main__":
    args = parse_args()
    generate_lidar_points(
        args.points_dir, args.calib_dir,
        args.output_dir, args.image_dir
    )
