SHELL := /bin/bash
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))
VERSION := $$(grep '^version' pyproject.toml | sed 's%version = "\(.*\)"%\1%')

.DEFAULT_GOAL := help

dev: ## setup development environment
	./setup.sh

help: ## This is help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
