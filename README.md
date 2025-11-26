# WIP Gentropy-manuscript

Repository to host code for the The Pleiotropic Map of Common Genetic Variation
and Therapeutic Implications.

## Chapters

Analysis conducted for each manuscript paragraphs are stored under consecutive
chapter subdirectory.

### 01-data preparation

The input datasets are downloaded and subsequent datasets are prepared for
downstream analysis.

## Running notebooks

To install all dependencies required to run notebooks run:

```{bash}
make dev
```

## Requirements

- Python 3.11+
- uv
- Java 11 (recommended using sdkman)
- ~40G of RAM
- ~30G of disk space
- gsutil & gcloud sdk (for downloading data from Google Cloud Storage)

## Adding new dependencies

To add new dependency use `uv add ${dependency}`

## Storing data files

All data stored for the purpose of the analysis should be dumped into the `data`
directory. They are not tracked by the git.
