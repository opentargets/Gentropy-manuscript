#!/usr/bin/env bash
# This script downloads the Epiraction dataset from ENCODE, processes it by unzipping, sorting,
# block gzipping, and indexing the files, and organizes them into a specified directory structure
# Usage: ./scripts/download-epiraction.sh
# NOTE: run from project root, otherwise data directory will not be found

set -euo pipefail
pwd

# Test for required commands
for cmd in bgzip tabix bedtools parallel uvx; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed." >&2
        echo "Please try 'brew install $cmd' or equivalent." >&2
        exit 1
    fi
done


echo "Starting"
ROOT_DIR=$(dirname $(dirname "$0"))

echo "$ROOT_DIR"
if [ ! -d "$ROOT_DIR/data" ]; then
    echo "data directory not found, try running the script from the project root"
    exit 1
fi


readonly DATA_DIR="$ROOT_DIR/data/25.06/EPIraction/raw"
readonly MANIFEST="$ROOT_DIR/data/epiraction-manifest.tsv"


if [ ! -f "$MANIFEST" ]; then
    echo "manifest file not found at $MANIFEST"
    exit 1
fi

if [  -d "$DATA_DIR" ]; then
    echo "data directory $DATA_DIR already exists, skipping"
    exit 0

else
    echo "Creating data directory at $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi


echo "Changing to data directory"

echo "Downloading Epiraction dataset from ENCODE"
time uvx --from git+https://github.com/project-defiant/encode-crawler@v0.1.0 crawler --output-dir "$DATA_DIR" --input "$MANIFEST" > /dev/null 2>&1
echo "Download complete."

echo "Unzipping"
time find "$DATA_DIR" -name 'ENC*.bed.gz' | parallel 'bgzip -d {}'
echo "Sorting"
time find "$DATA_DIR" -name 'ENC*.bed' | parallel "head -1 {} | sed '1s%#%%' > {.}.sorted.bed && bedtools sort -i {} >> {.}.sorted.bed"
echo "Block gzipping"
time find "$DATA_DIR" -name 'ENC*.sorted.bed' | parallel 'bgzip {}'
echo "Indexing"
time find "$DATA_DIR" -name 'ENC*.sorted.bed.gz' | parallel 'tabix -p bed -S 1 {}'
echo "Cleaning up intermediate files"
time find "$DATA_DIR" -name 'ENC*.bed' -type f -delete
echo "Completed."
