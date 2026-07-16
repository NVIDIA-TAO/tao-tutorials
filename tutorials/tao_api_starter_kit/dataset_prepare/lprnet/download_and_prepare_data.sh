#!/bin/bash
# usage:
#  bash download_and_prepare_data.sh /data-dir/openalpr_benchmark_dataset

set -e
set -x


if [ -z "$1" ]; then
  echo "usage download_and_preprocess_data.sh [data dir]"
  exit
fi


CURRENT_DIR=$(pwd)
echo "Cloning OpenALPR benchmark directory"
if [ ! -e benchmarks ]; then
  git clone https://github.com/openalpr/benchmarks benchmarks
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
mkdir -p "${OUTPUT_DIR}"

# Run our conversion
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "Preprocessing OpenALPR benchmarks data for US"
python3 $SCRIPT_DIR/preprocess_openalpr_benchmark.py \
    --input_dir=$CURRENT_DIR/benchmarks/endtoend/us/ \
    --output_dir=$OUTPUT_DIR \
