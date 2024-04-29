# Documentation Guide

## Installation

To get started with installation, run the following commands to install all doc-related dependencies with Poetry:

```sh
cd docs/
poetry install
```

Any additional documentation requirements should be added to the `docs/pyproject.toml` and the `requirements.txt` should be updated for the `.readthedocs.yaml` to correctly recognize all new dependencies.

To export poetry's dependencies to a `requirements.txt`, first install the `poetry-plugin-export` with `pipx`. 

```
pipx inject poetry poetry-plugin-export
```

Then run the following command to export the requirements to `requirements.txt` (ensure you're in the root `docs/` directory):
```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Getting Started

We use [MkDocs](https://www.mkdocs.org/) with [ReadTheDocs](https://about.readthedocs.com/?ref=readthedocs.com).

To run a local version of the documentation page (ensure you're in the root `docs/` directory):

```sh
mkdocs serve
```