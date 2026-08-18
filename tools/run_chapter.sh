#!/usr/bin/env bash
# Execute every notebook of a chapter in order, in place, stopping at the first failure.
#
#   tools/run_chapter.sh chapters/01-data-preparation
#   tools/run_chapter.sh chapters/02-analysis-main 03 04     # only these prefixes
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chapter="${1:?usage: run_chapter.sh <chapter-dir> [prefix ...]}"
shift || true

cd "$root"
for notebook in "$chapter"/*.ipynb; do
    name="$(basename "$notebook")"
    wanted=1
    if [ "$#" -gt 0 ]; then
        wanted=0
        for prefix in "$@"; do
            if [[ "$name" == "$prefix"* ]]; then
                wanted=1
            fi
        done
    fi
    if [ "$wanted" = 1 ]; then
        echo "=== $name"
        uv run jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=28800 "$notebook"
    fi
done
echo "=== chapter complete"
