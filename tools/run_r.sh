#!/usr/bin/env bash
# Run an R figure script against the project R library, from the repository root.
#
#   tools/run_r.sh chapters/04-figures-main/figure_2/figure_2.R
#
# Every figure script in chapters/04-figures-main and chapters/05-figures-supplementary
# expects the repository root as the working directory and reads its inputs from
# data/intermediate_files_refactor/.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# A bash glob, not `ls`: with CLICOLOR_FORCE set (Jupyter kernels do), `ls` wraps the path in ANSI
# escapes and R silently falls back to its default library.
shopt -s nullglob
libraries=("$root"/chapters/r-env/library/*/*/*)
lib="${libraries[0]-}"

if [ -z "$lib" ]; then
    echo "No R library found under chapters/r-env/library." >&2
    echo "Restore it with: R_LIBS_SITE= Rscript -e 'renv::restore(project=\"chapters/r-env\")'" >&2
    exit 1
fi

cd "$root"
# The repository .Rprofile activates a different renv project and would override the library
# path, so it is skipped.
R_PROFILE_USER=/dev/null R_LIBS_SITE="$lib" Rscript --no-init-file "$@"
