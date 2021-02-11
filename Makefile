#
# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=gensvm
DOC_DIR=./docs/
VENV_DIR=/tmp/gensvm_venv

.PHONY: help cover dist venv

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

################
# Installation #
################

.PHONY: inplace install

inplace: ## Build C extensions
	python setup.py build_ext -i

install: ## Install for the current user using the default python command
	python setup.py build_ext --inplace
	python setup.py install --user

################
# Distribution #
################

.PHONY: release dist

dist: ## Make Python source distribution
	python setup.py sdist

release: ## Prepare a release
	python make_release.py

###########
# Testing #
###########

.PHONY: test test_direct

test: venv ## Run nosetests using the default nosetests command
	source $(VENV_DIR)/bin/activate && green -a -vv -f ./tests

test_direct: inplace ## Run unit tests without a virtual environment
	pip install wheel && pip install . && \
		python -m unittest discover ./tests

#################
# Documentation #
#################

docs: doc
doc: venv ## Build documentation with Sphinx
	source $(VENV_DIR)/bin/activate && m2r README.md && mv README.rst $(DOC_DIR)
	source $(VENV_DIR)/bin/activate && m2r CHANGELOG.md && mv CHANGELOG.rst $(DOC_DIR)
	source $(VENV_DIR)/bin/activate && $(MAKE) -C $(DOC_DIR) html

#######################
# Virtual environment #
#######################

.PHONY: venv

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: setup.py
	test -d $(VENV_DIR) || python -m venv $(VENV_DIR)
	source $(VENV_DIR)/bin/activate && pip install -U numpy && \
		pip install -e .[dev]
	touch $(VENV_DIR)/bin/activate

############
# Clean up #
############

clean: ## Clean up after build or dist and remove compiled code
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./$(PACKAGE).egg-info
	rm -rf ./cover
	rm -rf $(VENV_DIR)
	rm -f MANIFEST
	rm -f ./$(PACKAGE)/cython_wrapper/*.so
	$(MAKE) -C ./src/gensvm clean
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

cleaner: clean ## Remove Cython output too
	rm -f ./src/wrapper.c
