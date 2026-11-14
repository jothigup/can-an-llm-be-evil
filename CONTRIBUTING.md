# Contributing

Thanks for contributing to this research project.

## Development setup

Use Python 3.10 or newer, create a virtual environment, and install the package
with its development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Before opening a change

Run the tests and lint check:

```bash
python -m pytest
python -m ruff check .
```

Add or update tests when changing behavior. Keep changes focused, avoid
committing credentials or private data, and document dataset/model-output
provenance and usage restrictions when relevant.

## Pull requests

Describe the motivation and summarize the change. Include test results and note
any data, model, or reproducibility considerations.
