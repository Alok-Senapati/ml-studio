.PHONY: install sync lint format test typecheck check clean

install:
	uv sync

sync:
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

typecheck:
	uv run mypy src

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src
	uv run pytest

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete