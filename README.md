# Gentropy Manuscript Analysis

Code repository for: **The Human Pleiotropic Map of GWAS Associations and Therapeutic Implications** ([preprint](https://www.biorxiv.org/content/10.64898/2026.04.28.721048v1)).

## System requirements

### Hardware
- RAM: ≥40 GB (required for data loading and analysis steps)
- Disk: ≥40 GB free space (for downloaded datasets and intermediate outputs)

### Operating system
- Linux (Ubuntu 20.04+) or macOS (12+)
- Windows is not supported

### Software prerequisites
| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥3.11, <3.14 | Core analysis |
| [uv](https://docs.astral.sh/uv/) | latest | Dependency management |
| Java | 11 (recommended via [sdkman](https://sdkman.io/)) | PySpark / Gentropy |
| R | ≥4.3 | Manuscript figures (Chapter 03) |
| [gcloud SDK](https://cloud.google.com/sdk/docs/install) | latest | Data download |
| gsutil | included with gcloud | Data download |

### Python dependencies (key packages)
Full pinned versions are in `uv.lock`. Core packages:

| Package | Version |
|---------|---------|
| gentropy | ≥2.4.1 |
| pandas | ≥2.2.0 |
| numpy | ≥1.26.4 |
| scipy | ≥1.11.4 |
| statsmodels | ≥0.14.4 |
| pymc | ≥5.12.0 |
| plotnine | 0.15.1 |
| blitzgsea | ≥0.5.0 |
| jupyterlab | ≥4.3.6 |

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/opentargets/Gentropy-manuscript.git
cd Gentropy-manuscript
```

### 2. Run the setup script
```bash
make dev
```
This installs `uv` (if not already present), selects the correct Python version, syncs all dependencies, and installs pre-commit hooks.

**Typical install time: ~10–15 minutes** on a standard desktop with a broadband connection.

### 3. Install R dependencies (for figure generation only)
```bash
cd chapters/03-manuscript-figures
Rscript -e "renv::restore()"
```
**Typical install time: ~10–15 minutes.**

### 4. Authenticate with Google Cloud
```bash
gcloud auth application-default login
```
Required to download processed datasets from the Open Targets release.

## Reproducing the analysis

The analysis is organised into three chapters, intended to be run sequentially.

### Chapter 01 — Data preparation
Downloads and processes input datasets.

```bash
uv run jupyter lab chapters/01-data-preparation/
```

Run notebooks in order:
1. `01_download_data_to_local_repo.ipynb` — downloads Open Targets release data (~40 GB; ~30–60 min depending on connection)
2. `02_lead_variant_effect_dataset_preparation.ipynb`
3. `03_the_list_of_replicated_CS.ipynb`
4. `04_qualifying_dataset_generation.ipynb`
5. `06_l2g_predictions.ipynb`
6. `07_gene_sets.ipynb`

**Expected output:** processed Parquet/TSV files written to `data/`.

### Chapter 02 — Analysis
Statistical analyses corresponding to each Results section. All notebooks can be run within ~2 hours on a machine meeting the hardware requirements above.

```bash
uv run jupyter lab chapters/02-analysis/
```

Sub-directories map to manuscript sections:
- `01-descriptions-numbers/` — panoramic overview statistics
- `02-variant-effects/` — selective pressures and variant effects
- `03-coloc-l2g/` — colocalisation and L2G
- `04-variant-level-ps/` — variant-level pleiotropy
- `05-gene-level-ps/` — gene-level pleiotropy
- `06-target-enrichment/` — clinical trial enrichment
- `07-pathway-enrichment/` — pathway analyses

**Expected output:** summary statistics tables and intermediate results written to `data/`.

### Chapter 03 — Manuscript figures
Generates all main and extended data figures. For per-figure instructions, including the mapping of panels to notebooks, scripts, and data files, see [FIGURE_MAPPING.md](FIGURE_MAPPING.md).

**Expected output:** PDF/PNG figures written to `chapters/03-manuscript-figures/figure_*/`.

## Repository structure

```
.
├── chapters/
│   ├── 01-data-preparation/   # Input data download and preprocessing
│   ├── 02-analysis/           # Statistical analyses (7 subdirectories)
│   └── 03-manuscript-figures/ # Figure generation (Python + R)
├── src/
│   └── manuscript_methods/    # Shared Python utilities
├── data/                      # Downloaded and intermediate data (not tracked)
├── pyproject.toml             # Python project and dependency specification
├── uv.lock                    # Pinned dependency versions
├── renv.lock                  # Pinned R dependency versions
├── Makefile                   # Development shortcuts
└── FIGURE_MAPPING.md          # Figure → code → data cross-reference
```

## License

Apache License 2.0
