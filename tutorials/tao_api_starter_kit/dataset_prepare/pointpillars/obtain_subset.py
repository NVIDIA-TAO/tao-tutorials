# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Obtain subset of pointpillars data"""
import argparse
import os
from tqdm import tqdm

"""
Usage:
python obtain_subset.py --source-data-dir=/home/user/data/training --out-data-dir=/home/user/subset_data/training/ --training True --num-images=100
python obtain_subset.py --source-data-dir=/home/user/data/testing --out-data-dir=/home/user/subset_data/testing/ --num-images=100
"""

def main():
    parser = argparse.ArgumentParser(description='Create a subset for kitti')
    parser.add_argument("--source-data-dir", type=str)
    parser.add_argument("--out-data-dir", type=str)
    parser.add_argument("--training", type=bool, default=False)
    parser.add_argument("--num-images", type=int)
    args = parser.parse_args()
    
    source_data_dir = args.source_data_dir
    out_data_dir = args.out_data_dir
    training_flag = bool(args.training)
    num_images = args.num_images

    # source_data_dir should contain the folders calib, image_2, label_2, velodyne
    if not os.path.exists(os.path.join(source_data_dir,"calib")):
        print("Download and extract kitti calib")
        exit()
    if not os.path.exists(os.path.join(source_data_dir,"image_2")):
        print("Download and extract kitti image_2")
        exit()
    if training_flag:
        if not os.path.exists(os.path.join(source_data_dir,"label_2")):
            print("Download and extract kitti label_2")
            exit()
    if not os.path.exists(os.path.join(source_data_dir,"velodyne")):
        print("Download and extract kitti velodyne")
        exit()

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    if not os.path.exists(os.path.join(out_data_dir,"calib")):
        os.makedirs(os.path.join(out_data_dir,"calib"))
    if not os.path.exists(os.path.join(out_data_dir,"image_2")):
        os.makedirs(os.path.join(out_data_dir,"image_2"))
    if training_flag:
      if not os.path.exists(os.path.join(out_data_dir,"label_2")):
          os.makedirs(os.path.join(out_data_dir,"label_2"))
    if not os.path.exists(os.path.join(out_data_dir,"velodyne")):
        os.makedirs(os.path.join(out_data_dir,"velodyne"))

    all_ids = os.listdir(os.path.join(source_data_dir,"image_2"))

    selected_ids = all_ids[:num_images]
    selected_ids = [id.replace('.png', '') for id in selected_ids]

    print(selected_ids)

    for id in tqdm(selected_ids):
        calib = source_data_dir + "/calib/" + str(id) + "*"
        os.system("cp " + calib + " " + out_data_dir + "/calib/")
        image = source_data_dir + "/image_2/" + str(id) + "*"
        os.system("cp " + image + " " + out_data_dir + "/image_2/")
        if training_flag:
          label = source_data_dir + "/label_2/" + str(id) + "*"
          os.system("cp " + label + " " + out_data_dir + "/label_2/")
        velodyne = source_data_dir + "/velodyne/" + str(id) + "*"
        os.system("cp " + velodyne + " " + out_data_dir + "/velodyne/")

if __name__ == "__main__":
    main()
