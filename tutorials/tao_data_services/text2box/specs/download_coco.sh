#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# Script to download and preprocess the COCO data set for detection.
#
# The outputs of this script are TFRecord files containing serialized
# tf.Example protocol buffers. See create_coco_tf_record.py for details of how
# the tf.Example protocol buffers are constructed and see
# http://cocodataset.org/#overview for an overview of the dataset.
#
# usage:
#  bash download_coco.sh /data-dir/coco
set -e
set -x


if [ -z "$1" ]; then
  echo "usage download_coco.sh [data dir]"
  exit
fi

#sudo apt install -y protobuf-compiler python-pil python-lxml\
#  python-pip python-dev git unzip

UNZIP="unzip -nq"

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

cd ${SCRATCH_DIR}

# Download the images.
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

TRAIN_IMAGE_FILE="train2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2017"

VAL_IMAGE_FILE="val2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
VAL_IMAGE_DIR="${SCRATCH_DIR}/val2017"

TEST_IMAGE_FILE="test2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TEST_IMAGE_FILE}
TEST_IMAGE_DIR="${SCRATCH_DIR}/test2017"

# Download the annotations.
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval2017.zip"
download_and_unzip ${BASE_INSTANCES_URL} ${INSTANCES_FILE}

TRAIN_OBJ_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_train2017.json"
VAL_OBJ_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_val2017.json"

TRAIN_CAPTION_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/captions_train2017.json"
VAL_CAPTION_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/captions_val2017.json"

# Download the test image info.
BASE_IMAGE_INFO_URL="http://images.cocodataset.org/annotations"
IMAGE_INFO_FILE="image_info_test2017.zip"
download_and_unzip ${BASE_IMAGE_INFO_URL} ${IMAGE_INFO_FILE}

TESTDEV_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/image_info_test-dev2017.json"

# mv ${SCRATCH_DIR}/annotations/ ${OUTPUT_DIR}
