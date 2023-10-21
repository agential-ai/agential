.PHONY: requirements clean lint help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = discussion-agents
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.10
POETRY_VERSION = 1.6.1

ifeq (,$(shell which conda 2>/dev/null))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements: test_environment ## Install Python dependencies with requirements.txt.
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

poetry_requirements: test_environment ## Install Python dependencies with Poetry.
	$(PYTHON_INTERPRETER) -m pip install pipx
	pipx install poetry=$(POETRY_VERSION)
	poetry check
	poetry check --lock
	poetry install

create_requirements: test_environment  ## Create requirements.txt (and dev) from pyproject.toml.
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --only=dev

clean: ## Delete all compiled Python files.
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint: ## Lint using black and ruff.
	poetry run mypy discussion_agents tests
	poetry run black --check discussion_agents tests
	poetry run ruff check discussion_agents tests

auto_lint: ## Automatic format & lint using black and ruff.
	poetry run black discussion_agents tests
	poetry run ruff discussion_agents tests --fix --show-fixes --show-source

test: ## Run all pytest tests.
	pytest tests/

test_nocost: ## Run pytest tests with no 'cost' marker (don't require funds to run).
	pytest -m "not cost" tests/

test_fast: ## Run pytest tests with no 'slow' marker and no 'cost' marker.
	pytest -m "not slow and not cost" tests/

create_environment: ## Set up conda environment.
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

test_environment: ## Test python environment is setup correctly.
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'
