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

"""Convert COCO annotations to ODVG format"""

import os
import json
import numpy as np

from pycocotools.coco import COCO
from tqdm.auto import tqdm


def xywh_to_xyxy(bbox):
    """Convert xywh to xyxy."""
    x, y, width, height = bbox
    x1 = round(x, 2)
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def clean_span(span):
    """Clean captions."""
    span = span.rstrip()
    span = span.replace('"', "'").replace('\"', "'").replace('“', "'").replace('”', "'")
    span = span.replace('‘', "'").replace('’', "'").replace('–', "—")
    if span.endswith('/') or span.endswith('.'):
        span = span[:-1]
    return span


def dump_label_map(cat_map, key_list, val_list, output):
    """Dump label mapping JSON file."""
    new_map = {}
    for key, value in cat_map.items():
        label = int(key)
        if label not in val_list:
            continue
        ind = val_list.index(label)
        label_trans = key_list[ind]
        new_map[label_trans] = value

    with open(output, "w", encoding="utf-8") as f:
        json.dump(new_map, f)


def convert_coco_to_odvg(coco_json_path, results_dir, use_all_categories=False, verbose=False):
    """Function to convert COCO annotations to ODVG format.

    Args:
        coco_json_path (str): Path to the COCO JSON file.
        results_dir (str): Path to the results directory.
        use_all_categories (bool): Whether to use all categories. Default is False.
        verbose (bool): verbosity. Default is False.
    """
    odvg_jsonl_path = os.path.join(results_dir, os.path.basename(coco_json_path).replace(".json", "_odvg.jsonl"))

    coco = COCO(coco_json_path)

    # check if the annotation is grounding dataset.
    if 'caption' in coco.imgs[list(coco.imgs.keys())[0]]:
        is_grounding = True
        print("Processing grounding annotations")
    else:
        is_grounding = False
        print("Processing detection annotations")

    # Dump categoy label mapping only if the annotation is detection only.
    if not is_grounding:
        cats = coco.loadCats(coco.getCatIds())
        names = {cat['id']: cat['name'] for cat in cats}

        if use_all_categories:
            cat_ids = list(names.keys())
            cat_cnts = [0] * len(cat_ids)
            for ann in coco.dataset['annotations']:
                cat_cnts[ann['category_id']] += 1
        else:
            # Get category ids that are actually present in the dataset.
            cat_ids, cat_cnts = np.unique([a['category_id'] for a in coco.dataset['annotations']], return_counts=True)

        if verbose:
            for cat_id, cat_cnt in zip(cat_ids, cat_cnts):
                print(f"{names[cat_id]} ({cat_id}): {cat_cnt}")

        # Now do the remapping of category ids to be contiguous.
        id_map = {}
        for idx, ci in enumerate(cat_ids):
            id_map[idx] = ci

        if verbose:
            print(f"Remapped total {len(names)} to {len(id_map)} so that class ids are contiguous")

        key_list = list(id_map.keys())
        val_list = list(id_map.values())

        dump_label_map(names, key_list, val_list, odvg_jsonl_path.replace(".jsonl", "_labelmap.json"))

    with open(odvg_jsonl_path, mode="w", encoding="utf-8") as writer:
        # iterate every annotation
        for img_id, img_info in tqdm(coco.imgs.items(), total=len(coco.imgs)):
            if is_grounding:
                caption = clean_span(img_info['caption'])

            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

            detection_list, grounding_list = [], []
            for ann in anns:
                bbox = ann['bbox']
                mask = ann.get('segmentation', None)
                bbox_xyxy = xywh_to_xyxy(bbox)
                if is_grounding:
                    token_positives = ann['tokens_positive']
                    phrase = ' '.join([caption[t[0]: t[1]] for t in token_positives])
                    grounding_annot = {
                        "bbox": bbox_xyxy,
                        "phrase": phrase,
                    }
                    if mask:
                        grounding_annot["mask"] = mask
                    grounding_list.append(grounding_annot)
                else:
                    label = ann['category_id']
                    category = names[label]
                    ind = val_list.index(label)
                    label_trans = key_list[ind]
                    dt_annot = {
                        "bbox": bbox_xyxy,
                        "label": label_trans,
                        "category": category
                    }
                    if mask:
                        dt_annot["mask"] = mask
                    detection_list.append(dt_annot)

            # For GoldG, skip if there are no annotations
            if len(anns) == 0 and len(grounding_list) == 0:
                if verbose:
                    print(f"Image ID {img_id} is being skipped.", caption)
                continue

            meta = {
                "file_name": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"]
            }

            if is_grounding:
                meta["grounding"] = {
                    "caption": caption,
                    "regions": grounding_list
                }
            else:
                meta["detection"] = {
                    "instances": detection_list
                }
            writer.write(f"{json.dumps(meta)}\n")

    print(f"ODVG annotation file is stored at {odvg_jsonl_path}")
