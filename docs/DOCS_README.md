# Documentation Guide

## Installation

To get started with installation, run the following commands to install all doc-related dependencies with Poetry:

```sh
cd docs/
poetry install
```

Any additional documentation requirements should be added to the `docs/pyproject.toml` and the `requirements.txt` should be updated for the `.readthedocs.yaml` to correctly recognize all new dependencies.

## Getting Started

We use [MkDocs](https://www.mkdocs.org/) with [ReadTheDocs](https://about.readthedocs.com/?ref=readthedocs.com).

To run a local version of the documentation page (ensure you're in the `docs/` directory):

```sh
mkdocs serve
```