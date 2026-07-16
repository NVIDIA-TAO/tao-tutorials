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
import pickle
import numpy as np

data_dir = os.path.join(os.environ["DATA_DIR"], "kinetics")

# front_raises: 134
# pull_ups: 255
# clean_and_jerk: 59
# presenting_weather_forecast: 254
# deadlifting: 88
selected_actions = {
    134: 0,
    255: 1,
    59: 2,
    254: 3,
    88: 4
}


def select_actions(selected_actions, data_dir, split_name):
    """Select a subset of actions and their corresponding labels.

    Args:
        selected_actions (dict): Map from selected class IDs to new class IDs.
        data_dir (str): Path to the directory of data arrays (.npy) and labels (.pkl).
        split_name (str): Name of the split to be processed, e.g., "train" and "val".

    Returns:
        No explicit returns
    """
    data_path = os.path.join(data_dir, f"{split_name}_data.npy")
    label_path = os.path.join(data_dir, f"{split_name}_label.pkl")

    data_array = np.load(file=data_path)
    with open(label_path, "rb") as label_file:
        labels = pickle.load(label_file)

    assert (len(labels) == 2)
    assert (data_array.shape[0] == len(labels[0]))
    assert (len(labels[0]) == len(labels[1]))

    print(f"No. total samples for {split_name}: {data_array.shape[0]}")

    selected_indices = []
    for i in range(data_array.shape[0]):
        if labels[1][i] in selected_actions.keys():
            selected_indices.append(i)

    data_array = data_array[selected_indices, :, :, :, :]
    selected_sample_names = [labels[0][x] for x in selected_indices]
    selected_labels = [selected_actions[labels[1][x]] for x in selected_indices]
    labels = (selected_sample_names, selected_labels)

    print(f"No. selected samples for {split_name}: {data_array.shape[0]}")

    np.save(file=data_path, arr=data_array, allow_pickle=False)
    with open(label_path, "wb") as label_file:
        pickle.dump(labels, label_file, protocol=4)


select_actions(selected_actions, data_dir, "train")
select_actions(selected_actions, data_dir, "val")
