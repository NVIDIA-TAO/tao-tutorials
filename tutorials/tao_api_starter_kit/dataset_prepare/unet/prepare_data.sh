#!/bin/bash
# usage:
#  bash prepare_data.sh /data-dir/isbi

set -e
set -x


if [ -z "$1" ]; then
  echo "usage prepare_data.sh [data dir]"
  exit
fi


# Create the output directories.
OUTPUT_DIR="${1%/}"

# Run our conversion
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "Preprocessing Unet benchmarks data for US"
python3 $SCRIPT_DIR/prepare_data_isbi.py \
    --input_dir=$OUTPUT_DIR \
    --output_dir=$OUTPUT_DIR \
