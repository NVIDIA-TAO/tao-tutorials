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

"""Convert COCO annotations to have contiguous class ids"""

import os
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


def convert_coco_to_contiguous(annotation_json_path, results_dir, use_all_categories=False, verbose=False):
    """Function to convert COCO to COCO Contiguous.

    Args:
        cfg (dataclas): Hydra Config.
        verbose (bool): verbosity. Default is False.
    """
    output_path = os.path.join(results_dir, os.path.basename(annotation_json_path).replace(".json", "_remapped.json"))

    coco = COCO(annotation_json_path)
    cats = coco.loadCats(coco.getCatIds())
    names = {cat['id']: cat['name'] for cat in cats}

    if use_all_categories:
        cat_ids = list(names.keys())
    else:
        # Get category ids that are actually present in the dataset.
        cat_ids, cat_cnts = np.unique([a['category_id'] for a in coco.dataset['annotations']], return_counts=True)
        for cat_id, cat_cnt in zip(cat_ids, cat_cnts):
            if verbose:
                print(f"{names[cat_id]} ({cat_id}): {cat_cnt}")

    # Now do the remapping of category ids to be contiguous.
    id_map = {}
    for idx, ci in enumerate(cat_ids):
        id_map[idx] = ci

    if verbose:
        print(f"Remapped total {len(names)} to {len(id_map)} so that classes are contiguous")

    key_list = list(id_map.keys())
    val_list = list(id_map.values())

    cats_list, anns_list = [], []
    for img_id in tqdm(coco.imgs.keys()):
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            label = ann['category_id']
            ind = val_list.index(label)
            label_trans = key_list[ind]
            ann['category_id'] = label_trans
            instance_list.append(ann)
        anns_list.extend(instance_list)

    for cat in coco.dataset["categories"]:
        label = cat['id']
        ind = val_list.index(label)
        label_trans = key_list[ind]
        cat['id'] = label_trans
        cats_list.append(cat)

    result = {
        "images": coco.dataset["images"],
        "annotations": anns_list,
        "categories": cats_list
    }

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(result, f)

    print(f"Remapped COCO json file is stored at {output_path}")
