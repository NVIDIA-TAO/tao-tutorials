#!/bin/bash
# Script can be used to download COCO dataset.
# usage:
#  bash download_coco.sh /data-dir
set -e
set -x


if [ -z "$1" ]; then
  echo "usage download_coco.sh [data dir]"
  exit
fi

UNZIP="unzip -nq"

# Create the output directories.
OUTPUT_DIR="${1%}/raw-data"
mkdir -p "${OUTPUT_DIR}"
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

cd ${OUTPUT_DIR}

# Download the images.
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

TRAIN_IMAGE_FILE="train2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
TRAIN_IMAGE_DIR="${OUTPUT_DIR}/train2017"

VAL_IMAGE_FILE="val2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
VAL_IMAGE_DIR="${OUTPUT_DIR}/val2017"

TEST_IMAGE_FILE="test2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TEST_IMAGE_FILE}
TEST_IMAGE_DIR="${OUTPUT_DIR}/test2017"

# Download the annotations.
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval2017.zip"
download_and_unzip ${BASE_INSTANCES_URL} ${INSTANCES_FILE}

TRAIN_OBJ_ANNOTATIONS_FILE="${OUTPUT_DIR}/annotations/instances_train2017.json"
VAL_OBJ_ANNOTATIONS_FILE="${OUTPUT_DIR}/annotations/instances_val2017.json"

TRAIN_CAPTION_ANNOTATIONS_FILE="${OUTPUT_DIR}/annotations/captions_train2017.json"
VAL_CAPTION_ANNOTATIONS_FILE="${OUTPUT_DIR}/annotations/captions_val2017.json"

# Download the test image info.
BASE_IMAGE_INFO_URL="http://images.cocodataset.org/annotations"
IMAGE_INFO_FILE="image_info_test2017.zip"
download_and_unzip ${BASE_IMAGE_INFO_URL} ${IMAGE_INFO_FILE}

TESTDEV_ANNOTATIONS_FILE="${OUTPUT_DIR}/annotations/image_info_test-dev2017.json"