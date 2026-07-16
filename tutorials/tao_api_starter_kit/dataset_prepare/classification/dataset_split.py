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

from os.path import join as join_path
import os
import glob
import re
import shutil
import json

DATA_DIR=os.environ.get('DATA_DIR')
source_dir = join_path(DATA_DIR, "VOCdevkit/VOC2012")
target_dir = join_path(DATA_DIR, "formatted")

suffix = '_trainval.txt'
classes_dir = join_path(source_dir, "ImageSets", "Main")
images_dir = join_path(source_dir, "JPEGImages")
classes_files = glob.glob(classes_dir+"/*"+suffix)
class_names = []
for file in classes_files:
    # get the filename and make output class folder
    classname = os.path.basename(file)
    if classname.endswith(suffix):
        classname = classname[:-len(suffix)]
        target_dir_path = join_path(target_dir, classname)
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
    else:
        continue
    print(classname)
    class_names.append(classname)

    with open(file) as f:
        content = f.readlines()

    for line in content:
        tokens = re.split('\s+', line)
        if tokens[1] == '1':
            # copy this image into target dir_path
            target_file_path = join_path(target_dir_path, tokens[0] + '.jpg')
            src_file_path = join_path(images_dir, tokens[0] + '.jpg')
            shutil.copyfile(src_file_path, target_file_path)

from random import shuffle
from tqdm import tqdm

DATA_DIR=os.environ.get('DATA_DIR')
SOURCE_DIR=os.path.join(DATA_DIR, 'formatted')
TARGET_DIR=os.path.join(DATA_DIR,'split')
# list dir
print(os.walk(SOURCE_DIR))
dir_list = next(os.walk(SOURCE_DIR))[1]
# for each dir, create a new dir in split
for dir_i in tqdm(dir_list):
    newdir_train = os.path.join(TARGET_DIR, 'images_train', dir_i)
    newdir_val = os.path.join(TARGET_DIR, 'images_val', dir_i)
    newdir_test = os.path.join(TARGET_DIR, 'images_test', dir_i)

    if not os.path.exists(newdir_train):
            os.makedirs(newdir_train)
    if not os.path.exists(newdir_val):
            os.makedirs(newdir_val)
    if not os.path.exists(newdir_test):
            os.makedirs(newdir_test)

    img_list = glob.glob(os.path.join(SOURCE_DIR, dir_i, '*.jpg'))
    # shuffle data
    shuffle(img_list)

    for j in range(int(len(img_list)*0.7)):
            shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'images_train', dir_i))

    for j in range(int(len(img_list)*0.7), int(len(img_list)*0.8)):
            shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'images_val', dir_i))

    for j in range(int(len(img_list)*0.8), len(img_list)):
            shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'images_test', dir_i))

with open(f'{DATA_DIR}/classes.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

classmap = {}
for idx, class_name in enumerate(class_names):
    classmap[class_name] = idx

with open(f'{DATA_DIR}/classmap.json', 'w') as json_file:
    json.dump(classmap, json_file)

shutil.copy2(f'{DATA_DIR}/classes.txt',TARGET_DIR)
shutil.copy2(f'{DATA_DIR}/classmap.json',TARGET_DIR)
print('Done splitting dataset.')
