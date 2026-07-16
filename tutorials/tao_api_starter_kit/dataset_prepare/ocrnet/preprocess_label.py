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

# Clean the label to alphanumeric, non-sensitive (lower case). Filter the label with length larger than 25
import os
import re
import sys
from tqdm import tqdm

def preprocess_label(gt_file, filtered_file):
    gt_list = open(gt_file, "r").readlines()
    filtered_list = []

    character_list = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for label_line in tqdm(gt_list):
        try:
            path, label = label_line.strip().split()
        except Exception:
            continue
        path = path[:-1]
        label = label.strip("\"")
        if re.search(f"[^{character_list}]", label):
            continue
        else:
            if len(label) <= 25:
                label = label.lower() # ignore the case
                filtered_list.append(f"{path}\t{label}\n")

    with open(filtered_file, "w") as f:
        f.writelines(filtered_list)

def main():
    preprocess_label(sys.argv[1], sys.argv[2])
    character_list = "0123456789abcdefghijklmnopqrstuvwxyz"
    with open(os.path.join(os.getenv("DATA_DIR"), "character_list"), "w") as f:
         for ch in character_list:
                f.write(f"{ch}\n")

if __name__ == "__main__":
    main()