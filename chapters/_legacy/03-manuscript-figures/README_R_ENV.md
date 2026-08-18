# R Environment Setup

This directory uses `renv` for R package management to ensure reproducible
analysis.

## Setup

The R environment has been initialized with all required packages. When you open
R in this directory, `renv` will automatically activate.

## Required Packages

The following packages are installed in this environment:

- **Visualization**: `circlize`, `ggplot2`, `cowplot`, `patchwork`,
  `RColorBrewer`, `png`
- **Data manipulation**: `dplyr`, `tidyr`, `readr`, `arrow`
- **Utilities**: `scales`, `stringr`, `rlang`, `rstudioapi`

## Usage

### Running R scripts

Simply run your R scripts as usual. The environment will be automatically
activated:

```r
source("figure_1/manh_plot.R")
```

### Installing new packages

If you need to install additional packages:

```r
renv::install("package_name")
renv::snapshot()  # Update the lockfile
```

### Restoring the environment

If you need to restore packages from the lockfile (e.g., on a new machine):

```r
renv::restore()
```

### Checking status

To see the current status of packages:

```r
renv::status()
```

## Files

- `renv.lock` - Lockfile containing all package versions (commit this to version
  control)
- `.Rprofile` - Automatically activates renv when R starts in this directory
- `renv/` - Directory containing the local R library and renv configuration

## Notes

- The `renv` library is stored locally in `renv/library/`
- Package versions are locked to ensure reproducibility
- The lockfile should be committed to version control
