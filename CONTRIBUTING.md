# Contributing to vczstore

## Development setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/tomwhite/vczstore.git
cd vczstore
uv sync --group dev
```

## Running tests

```bash
uv run pytest
```

Note: some tests require `bcftools` and `plink` to be installed
(available via conda-forge and bioconda).

## Linting

We use [prek](https://github.com/prek-dev/prek) for pre-commit linting,
configured in `prek.toml`. Install it as a pre-commit hook:

```bash
uv run prek install
```

Run all checks manually:

```bash
uv run --only-group=lint prek -c prek.toml run --all-files
```

If local results differ from CI, run `uv run prek cache clean`.

## Pull requests

- Create a branch from `main`
- Ensure all CI checks pass
- Add a changelog entry if appropriate
