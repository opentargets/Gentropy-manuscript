SHELL := /bin/bash
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))
VERSION := $$(grep '^version' pyproject.toml | sed 's%version = "\(.*\)"%\1%')

.DEFAULT_GOAL := help

dev: ## setup development environment
	./setup.sh

help: ## This is help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


upload_to_gcs: ## Upload created dataset to google cloud storage
	@gcloud storage cp data/rescaled-betas.parquet gs://genetics-portal-dev-analysis/ss60/gentropy-manuscript/chapters/variant-effect-prediction/rescaled-betas.parquet
	@gcloud storage cp data/binary-therapeutic-lead-variants.parquet gs://genetics-portal-dev-analysis/ss60/gentropy-manuscript/chapters/variant-effect-prediction/binary-therapeutic-lead-variants.parquet

download_from_gcs: ## Download datasets
	@gcloud storage rsync -r gs://genetics-portal-dev-analysis/yt4/20250403_for_gentropy_paper/gwas_study_index_with_theraputic_areas data/therapeutic_areas
	@gcloud storage cp gs://genetics-portal-dev-analysis/ss60/gentropy-manuscript/chapters/variant-effect-prediction/rescaled-betas.parquet data/rescaled-betas.parquet
	@gcloud storage cp gs://genetics-portal-dev-analysis/ss60/gentropy-manuscript/chapters/variant-effect-prediction/binary-therapeutic-lead-variants.parquet data/binary-therapeutic-lead-variants.parquet
	@gcloud storage rsync -r gs://genetics-portal-dev-analysis/yt4/20250403_for_gentropy_paper/known_studyLocusIds data/known_studyLocusIds
