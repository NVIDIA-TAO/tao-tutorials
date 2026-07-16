# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

import os
import argparse

import numpy as np

from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils.object3d_kitti import (
    get_objects_from_label
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils.calibration_kitti import (
    Calibration
)

def parse_args():
    parser = argparse.ArgumentParser("Convert camera label to LiDAR label.")
    parser.add_argument(
        "-l", "--label_dir",
        type=str, required=True,
        help="Camera label directory."
    )
    parser.add_argument(
        "-c", "--calib_dir",
        type=str, required=True,
        help="Calibration file directory"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str, required=True,
        help="Output LiDAR label directory"
    )
    return parser.parse_args()


def generate_lidar_labels(label_dir, calib_dir, output_dir):
    """Generate LiDAR labels from KITTI Camera labels."""
    for lab in os.listdir(label_dir):
        lab_file = os.path.join(label_dir, lab)
        obj_list = get_objects_from_label(lab_file)
        calib_file = os.path.join(calib_dir, lab)
        calib = Calibration(calib_file)
        loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        loc_lidar = calib.rect_to_lidar(loc)
        # update obj3d.loc
        with open(os.path.join(output_dir, lab), "w") as lf:
            for idx, lc in enumerate(loc_lidar):
                # bottom center to 3D center
                obj_list[idx].loc = (lc + np.array([0., 0., obj_list[idx].h / 2.]))
                # rotation_y to rotation_z
                obj_list[idx].ry = -np.pi / 2. - obj_list[idx].ry
                lf.write(obj_list[idx].to_kitti_format())
                lf.write('\n')


if __name__ == "__main__":
    args = parse_args()
    generate_lidar_labels(args.label_dir, args.calib_dir, args.output_dir)
