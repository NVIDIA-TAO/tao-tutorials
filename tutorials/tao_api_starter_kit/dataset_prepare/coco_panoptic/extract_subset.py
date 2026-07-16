# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import random
import json
import sys


"""Extract a subset of images from COCO dataset"""
"""Usage: python3 extract_subset.py <images_path> <masks_path> <instances_path> <panoptic_path> <output_dir> <num_images>"""

if len(sys.argv) < 4:
    print(f"Usage: python3 extract_subset.py <images_path> <masks_path> <instances_path> <panoptic_path> <output_dir> <num_images>")
    exit(1)

images_path = sys.argv[1]
masks_path = sys.argv[2]
instances_path = sys.argv[3]
panoptic_path = sys.argv[4]
output_dir = sys.argv[5]
num_images = int(sys.argv[6]) if len(sys.argv) > 6 else 100 # default random extract 100 images

print(f"Extracting {num_images} images from {images_path} and saving to {output_dir}")

# Load annotations
with open(instances_path, 'r') as f:
    instances_annotations = json.load(f)
with open(panoptic_path, 'r') as f:
    panoptic_annotations = json.load(f)

# Randomly select num_images image IDs
image_ids = [img['id'] for img in instances_annotations['images']]
selected_ids = random.sample(image_ids, num_images)

# copy the selected images to the output directory
selected_image_paths = []
for img in instances_annotations['images']:
    if img['id'] in selected_ids:
        selected_image_paths.append(img['file_name'])

os.makedirs(f"{output_dir}/images", exist_ok=True)
for file_name in selected_image_paths:
    src_path = os.path.join(images_path, file_name)
    dst_path = os.path.join(f"{output_dir}/images", file_name)
    os.system(f'cp {src_path} {dst_path}')

# copy the selected masks to the output directory
selected_mask_paths = []
for img in panoptic_annotations['images']:
    if img['id'] in selected_ids:
        selected_mask_paths.append(img['file_name'].replace('.jpg', '.png'))

os.makedirs(f"{output_dir}/masks", exist_ok=True)
for file_name in selected_mask_paths:
    src_path = os.path.join(masks_path, file_name)
    dst_path = os.path.join(f"{output_dir}/masks", file_name)
    os.system(f'cp {src_path} {dst_path}')

# extract instances annotations for the selected images
selected_annotations = []
for ann in instances_annotations['annotations']:
    if ann['image_id'] in selected_ids:
        selected_annotations.append(ann)
# Save the selected annotations to a new JSON file while keeping original format
selected_images = [img for img in instances_annotations['images'] if img['id'] in selected_ids]
selected_categories = instances_annotations['categories']
output_annotations = {
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': selected_categories
}

with open(os.path.join(output_dir, os.path.basename(instances_path)), 'w') as f:
    json.dump(output_annotations, f)

# extract panoptic annotations for the selected images
selected_annotations = []
for ann in panoptic_annotations['annotations']:
    if ann['image_id'] in selected_ids:
        selected_annotations.append(ann)
# Save the selected annotations to a new JSON file while keeping original format
selected_images = [img for img in panoptic_annotations['images'] if img['id'] in selected_ids]
selected_categories = panoptic_annotations['categories']
output_annotations = {
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': selected_categories
}

with open(os.path.join(output_dir, os.path.basename(panoptic_path)), 'w') as f:
    json.dump(output_annotations, f)
