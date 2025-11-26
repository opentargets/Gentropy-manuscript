# Gentropy-manuscript

Repository to host code for the Systematic and scalable analysis of common
variation advances drug target discovery

## Chapters

Analysis conducted for each manuscript paragraphs are stored under consecutive
chapter subdirectory.

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

## Adding new dependencies

To add new dependency use `uv add ${dependency}`

## Storing data files

All data stored for the purpose of the analysis should be dumped into the `data`
directory. They are not tracked by the git.
