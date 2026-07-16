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
"""Usage: python3 extract_subset.py <images_path> <annotation_path> <output_dir> <num_images> <is_val>"""

if len(sys.argv) < 4:
    print(f"Usage: python3 extract_subset.py <images_path> <annotation_path> <output_dir> <num_images> <is_val>")
    exit(1)

images_path = sys.argv[1]
annotation_path = sys.argv[2]
output_dir = sys.argv[3]
is_val = sys.argv[4] in ["1", "True", "true", "TRUE"] if len(sys.argv) > 4 else False
num_images = int(sys.argv[5]) if len(sys.argv) > 5 else 100 # default random extract 100 images

print(f"Extracting {num_images} images from {images_path} and saving to {output_dir}")

# Load annotations
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

# Get all image IDs
image_ids = [img['id'] for img in annotations['images']]

# Randomly select num_images image IDs
selected_ids = random.sample(image_ids, num_images)

# copy the selected images to the output directory
selected_image_paths = []
for img in annotations['images']:
    if img['id'] in selected_ids:
        selected_image_paths.append(img['file_name'])

os.makedirs(f"{output_dir}/images", exist_ok=True)
for file_name in selected_image_paths:
    src_path = os.path.join(images_path, file_name)
    dst_path = os.path.join(f"{output_dir}/images", file_name)
    os.system(f'cp {src_path} {dst_path}')

# extract annotations for the selected images
selected_annotations = []
for ann in annotations['annotations']:
    if ann['image_id'] in selected_ids:
        selected_annotations.append(ann)
# Save the selected annotations to a new JSON file while keeping original format
selected_images = [img for img in annotations['images'] if img['id'] in selected_ids]
selected_categories = annotations['categories']
output_annotations = {
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': selected_categories
}

with open(os.path.join(output_dir, os.path.basename(annotation_path)), 'w') as f:
    json.dump(output_annotations, f)

# extract label_map if is_val is True
if is_val:
    label_map = {}
    for ann in annotations['annotations']:
        if ann['image_id'] in selected_ids:
            label_map[ann['category_id']] = ann['id']
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
