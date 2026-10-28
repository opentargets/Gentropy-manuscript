# Data download

`01_download_data_to_local_repo.ipynb` fetches everything the pipeline reads
into `data/`:

- the Open Targets 25.06 release, from `gs://open-targets-data-releases/25.06`
  (requester-pays: set `BILLING_PROJECT`; also free over HTTPS from the EBI
  FTP). Requires `gcloud auth login`. ~24 GB, 30–60 min;
- the project inputs, from Zenodo
  ([10.5281/zenodo.23736324](https://doi.org/10.5281/zenodo.23736324), all
  versions), 206 MB. Add `intermediates.tar.gz` to skip chapter 01;
- one third-party file, Supplementary Table 8 of Sun et al. 2024, which has to
  be downloaded by hand. The notebook checks it is present.
