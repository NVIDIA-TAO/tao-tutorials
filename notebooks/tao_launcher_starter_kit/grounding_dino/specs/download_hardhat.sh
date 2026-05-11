#!/bin/bash
# Script can be used to download Hard Hat dataset.
# usage:
#  bash download_hardhat.sh /data-dir
set -e
set -x

if [ -z "$1" ]; then
  echo "usage download_hardhat.sh [data dir]"
  exit
fi

UNZIP="unzip -nq"

BASE_URL="https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35/HardHatWorkers.zip"

# Create the output directories.
OUTPUT_DIR="${1%}"
mkdir -p "${OUTPUT_DIR}"

cd ${OUTPUT_DIR}

wget ${BASE_URL} -O "hardhat.zip"

echo "Unzipping ${BASE_URL}"

${UNZIP} "hardhat.zip" -d ${OUTPUT_DIR}

rm "${OUTPUT_DIR}/hardhat.zip"
