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

import os
import sys
import cv2
import csv
import ujson


classes = set([])


def read_kitti(prefix, label_file):
    "Function wrapper to read kitti format labels txt file."
    global classes
    full_label_path = os.path.join(prefix, label_file)
    if not full_label_path.endswith(".txt"):
        return
    if not os.path.exists(full_label_path):
        raise ValueError("Labelfile : {} does not exist".format(full_label_path))
    if os.path.isdir(full_label_path):
        return

    dict_list = []

    image_name = full_label_path.replace("/labels", "/images").replace(".txt", ".jpg")
    if not os.path.exists(image_name):
        raise ValueError("Image  : {} does not exist".format(image_name))
    img = cv2.imread(image_name, 0)
    height, width = img.shape[:2]

    with open(full_label_path, 'r') as lf:
        for row in csv.reader(lf, delimiter=' '):
            classes.add(row[0])
            dict_list.append({"class_name": row[0],
                              "file_name": label_file.replace(".txt", ".jpg"),
                              "height": height,
                              "width": width,
                              "bbox": [float(row[4]), float(row[5]), float(row[6]) - float(row[4]), float(row[7]) - float(row[5])]})
    if (dict_list == []):
        dict_list = [{"file_name": label_file.replace(".txt", ".jpg"),
                      "height": height,
                      "width": width}]

    return dict_list


def construct_coco_json(labels_folder):
    image_id = 0
    annot_ctr = 0

    labels = []
    for file in os.listdir(labels_folder):
        label = read_kitti(labels_folder, file)
        labels.append(label)

    categories = []
    class_to_id_mapping = {}
    for idx, object_class in enumerate(classes):
        class_to_id_mapping[object_class] = idx + 1
        categories.append({"supercategory": object_class, "id": idx + 1, "name": object_class})
    coco_json = {"images": [], "annotations": [], "categories": categories}

    for label in labels:
        if not (label and len(label)):
            continue
        coco_json["images"].append({"file_name": label[0]["file_name"], "height": label[0]["height"], "width": label[0]["width"], "id": image_id})
        for instance in label:
            if ("bbox" in instance.keys()):
                coco_json["annotations"].append({"bbox": instance["bbox"],
                                                 "image_id": image_id,
                                                 "id": annot_ctr,
                                                 "category_id": class_to_id_mapping[instance["class_name"]],
                                                 "bbox_mode": 1,
                                                 "segmentation": [],
                                                 "iscrowd": 0,
                                                 "area": float(instance["bbox"][2] * instance["bbox"][3])})
                annot_ctr += 1
        image_id += 1
    return coco_json


label_folder = sys.argv[1]
coco_json = construct_coco_json(label_folder)

current_str = ujson.dumps(coco_json, indent=4)
with open(sys.argv[2] + "/annotations.json", "w") as json_out_file:
    json_out_file.write(current_str)

label_map_extension = sys.argv[3]
with open(f"{sys.argv[2]}/label_map.{label_map_extension}", "w") as label_map_file:
    for idx, class_name in enumerate(classes):
        if label_map_extension == "yaml":
            label_map_file.write(f"{idx+1}: '{class_name}'\n")
        else:
            label_map_file.write(f"{class_name}\n")
        label_map_file.flush()

print(len(classes))
