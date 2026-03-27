.PHONY: lint format check typecheck test coverage all clean lint-path format-path install reinstall docs docs-serve hooks

# Run all checks (lint + typecheck)
all: lint typecheck

# Lint with ruff (check only)
lint:
	uv run ruff check src/ tests/

# Lint a specific path: make lint-path PATH=src/motifnet/stochastic.py
lint-path:
	uv run ruff check $(PATH)

# Format with ruff
format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/
	uv run ty src/


# Format a specific path: make format-path PATH=src/motifnet/stochastic.py
format-path:
	uv run ruff format $(PATH)
	uv run ruff check --fix $(PATH)

# Run tests
test:
	uv run pytest tests/

# Run tests with coverage report
coverage:
	uv run pytest tests/ --cov=src/ 
	rm .coverage

# Install git hooks
hooks:
	git config core.hooksPath .githooks

# Install dependencies and hooks
install:
	uv sync
	$(MAKE) hooks

# Reinstall from scratch (clean venv + lockfile, then install)
reinstall:
	uv sync --reinstall

# Build documentation
docs:
	uv run mkdocs build --strict

# Serve documentation locally with live reload
docs-serve:
	uv run mkdocs serve

# Clean up cache files
clean:
	rm -rf .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
