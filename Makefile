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

in: inplace
inplace:
	python setup.py build_ext -i

install: ## Install for the current user using the default python command
	python setup.py build_ext --inplace
	python setup.py install --user

test: venv ## Run nosetests using the default nosetests command
	source $(VENV_DIR)/bin/activate && green -a -vv -f

develop: ## Install a development version of the package needed for testing
	python setup.py develop --user

dist: ## Make Python source distribution
	python setup.py sdist

docs: doc
doc: venv ## Build documentation with Sphinx
	source $(VENV_DIR)/bin/activate && m2r README.md && mv README.rst $(DOC_DIR)
	source $(VENV_DIR)/bin/activate && m2r CHANGELOG.md && mv CHANGELOG.rst $(DOC_DIR)
	source $(VENV_DIR)/bin/activate && $(MAKE) -C $(DOC_DIR) html

clean: ## Clean build dist and egg directories left after install
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

cleaner: clean
	rm -f ./src/wrapper.c
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	test -d $(VENV_DIR) || virtualenv $(VENV_DIR)
	source $(VENV_DIR)/bin/activate && pip install numpy && pip install -e .[dev]
	touch $(VENV_DIR)/bin/activate
