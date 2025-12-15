#!/usr/bin/env bash

set -euo pipefail
pwd

# Test for required commands
for cmd in curl bgzip tabix bedtools parallel; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed." >&2
        exit 1
    fi
done


echo "Starting"
DIR=$(dirname "$0")
if [ ! -d "$DIR/data" ]; then
    echo "data directory not found, try running the script from the project root"
    exit 1
fi

readonly DATA_DIR="$DIR/data/intermediate_files/"
readonly MANIFEST="$DIR/manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "manifest file not found at $MANIFEST"
    exit 1
fi


echo "Changing to data directory"
cd "$DATA_DIR" || exit 1

echo "Downloading Epiraction dataset from ENCODE"
xargs -n 1 curl -O -L < "$MANIFEST"
echo "Download complete."

echo "Unzipping"
find . -name 'ENC*.bed.gz' | parallel 'bgzip -d {}'
echo "Sorting"
find . -name 'ENC*.bed' | parallel 'bedtools sort -i {} > {.}.sorted.bed'
echo "Block gzipping"
find . -name 'ENC*.sorted.bed' | parallel 'bgzip {}'
echo "Indexing"
find . -name 'ENC*.sorted.bed.gz' | parallel 'tabix -p bed {}'
echo "Cleaning up intermediate files"
find . -name 'ENC*.bed' -type f -delete
echo "Completed."
