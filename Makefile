# Makefile for Finite Difference Computing with PDEs book

.PHONY: pdf html all preview clean test test-devito test-no-devito lint format check help

# Default target
all: pdf

# Build targets
pdf:
	quarto render --to pdf

html:
	quarto render --to html

# Build both PDF and HTML
book:
	quarto render

# Live preview with hot reload
preview:
	quarto preview

# Clean build artifacts
clean:
	rm -rf _book/
	rm -rf .quarto/
	find . -name "*.aux" -delete
	find . -name "*.log" -delete
	find . -name "*.out" -delete

# Test targets
test:
	pytest tests/ -v

test-devito:
	pytest tests/ -v -m devito

test-no-devito:
	pytest tests/ -v -m "not devito"

test-phase1:
	pytest tests/test_elliptic_devito.py tests/test_burgers_devito.py tests/test_swe_devito.py -v

# Linting and formatting
lint:
	ruff check src/

format:
	ruff check --fix src/
	isort src/

check:
	pre-commit run --all-files

# Help
help:
	@echo "Available targets:"
	@echo "  pdf          - Build PDF (default)"
	@echo "  html         - Build HTML"
	@echo "  book         - Build all formats (PDF + HTML)"
	@echo "  preview      - Live preview with hot reload"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run all tests"
	@echo "  test-devito  - Run only Devito tests"
	@echo "  test-no-devito - Run tests without Devito"
	@echo "  test-phase1  - Run Phase 1 tests (elliptic, burgers, swe)"
	@echo "  lint         - Check code with ruff"
	@echo "  format       - Auto-format code with ruff and isort"
	@echo "  check        - Run all pre-commit hooks"
	@echo "  help         - Show this help message"
