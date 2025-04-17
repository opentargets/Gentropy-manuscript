# Gentropy-manuscript

Repository to host code for the Systematic and scalable analysis of common
variation advances drug target discovery

## Chapters

Analysis conducted for each manuscript paragraphs are stored under consecutive
chapter subdirectory.

## Running notebooks

To install all dependencies required to run notebooks run:

```
make dev
```

## Adding new dependencies

To add new dependency use `uv add ${dependency}`

## Storing data files

All data stored for the purpose of the analysis should be dumped into the `data`
directory. They are not tracked by the git.
