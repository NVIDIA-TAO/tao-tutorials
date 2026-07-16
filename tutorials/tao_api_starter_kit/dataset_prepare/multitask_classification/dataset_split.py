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
import shutil
import numpy as np
import pandas as pd

DATA_DIR=os.environ.get('DATA_DIR')
df = pd.read_csv(os.environ['DATA_DIR'] + '/styles.csv', on_bad_lines='skip')
df = df[['id', 'baseColour', 'subCategory', 'season']]
df = df.dropna()
category_cls = df.subCategory.value_counts()[:10].index # 10-class multitask-classification
season_cls = ['Spring', 'Summer', 'Fall', 'Winter'] # 4-class multitask-classification
color_cls = df.baseColour.value_counts()[:11].index # 11-class multitask-classification

# Get all valid rows
df = df[df.subCategory.isin(category_cls) & df.season.isin(season_cls) & df.baseColour.isin(color_cls)]
df.columns = ['fname', 'base_color', 'category', 'season']
df.fname = df.fname.astype(str)
df.fname = df.fname + '.jpg'

# remove entries whose image file is missing
all_img_files = os.listdir(os.environ['DATA_DIR'] + '/images')
df = df[df.fname.isin(all_img_files)]

idx = np.arange(len(df))
np.random.shuffle(idx)

train_split_idx = int(len(df)*0.8)
train_df = df.iloc[idx[:train_split_idx]]
val_df = df.iloc[idx[train_split_idx:train_split_idx+(len(df) // 10)]]
test_df = df.iloc[idx[train_split_idx+(len(df) // 10):]]

# Add a simple sanity check
assert len(train_df.season.unique()) == 4 and len(train_df.base_color.unique()) == 11 and \
    len(train_df.category.unique()) == 10, 'Training set misses some classes, re-run this cell!'
assert len(val_df.season.unique()) == 4 and len(val_df.base_color.unique()) == 11 and \
    len(val_df.category.unique()) == 10, 'Validation set misses some classes, re-run this cell!'
assert len(test_df.season.unique()) == 4 and len(test_df.base_color.unique()) == 11 and \
    len(test_df.category.unique()) == 10, 'Test set misses some classes, re-run this cell!'

for image_name in train_df["fname"]:
    source_file_name = os.path.join(DATA_DIR, "images", image_name)
    destination_file_name = os.path.join(DATA_DIR, "images_train", image_name)
    shutil.copy(source_file_name, destination_file_name)

for image_name in val_df["fname"]:
    source_file_name = os.path.join(DATA_DIR, "images", image_name)
    destination_file_name = os.path.join(DATA_DIR, "images_train", image_name)
    shutil.copy(source_file_name, destination_file_name)
    destination_file_name = os.path.join(DATA_DIR, "images_val", image_name)
    shutil.copy(source_file_name, destination_file_name)

for image_name in test_df["fname"]:
    source_file_name = os.path.join(DATA_DIR, "images", image_name)
    destination_file_name = os.path.join(DATA_DIR, "images_test", image_name)
    shutil.copy(source_file_name, destination_file_name)

# save processed data labels
train_df.to_csv(os.environ['DATA_DIR'] + '/train.csv', index=False)
val_df.to_csv(os.environ['DATA_DIR'] + '/val.csv', index=False)