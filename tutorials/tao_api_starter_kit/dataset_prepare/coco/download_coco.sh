#!/bin/bash
# Script to download and preprocess the COCO data set
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

# Download the annotations.
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval2017.zip"
download_and_unzip ${BASE_INSTANCES_URL} ${INSTANCES_FILE}