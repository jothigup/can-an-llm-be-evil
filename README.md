# Can an LLM Be Evil?

A research project for studying harmful or undesirable behavior in large language
models. The repository is set up as an installable Python package; research code,
experiments, and findings can be added as the project develops.

## Getting started

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite and lint checks:

```bash
python -m pytest
python -m ruff check .
```

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── src/can_an_llm_be_evil/   # Installable package
├── tests/                    # Automated tests
├── CONTRIBUTING.md
└── pyproject.toml            # Package and development-tool configuration
```

## Research and data

Add experiments and analysis as focused modules under `src/can_an_llm_be_evil/`,
with tests under `tests/`. Keep credentials and private data out of the
repository. Document the source, license, and handling requirements for any
datasets or model outputs before adding them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.
