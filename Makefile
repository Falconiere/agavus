# Makefile recipes require literal tabs; all other files use 2 spaces.
.PHONY: install lint format typecheck dupcheck depcheck sizes test check hooks

install:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check src scripts space
	uv run ruff format --check src scripts space

format:
	uv run ruff format src scripts space
	uv run ruff check --fix src scripts space

typecheck:
	uv run mypy

dupcheck:
	uv run python scripts/check_duplication.py src scripts space

depcheck:
	uv run deptry src space

sizes:
	uv run python scripts/check_sizes.py src scripts space

test:
	uv run pytest

check: lint typecheck dupcheck depcheck sizes test

hooks:
	uv run pre-commit run --all-files