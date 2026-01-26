# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source repository for *Finite Difference Computing with PDEs - A Modern Software Approach* by Hans Petter Langtangen and Svein Linge. The book teaches finite difference methods for solving PDEs through Python implementations.

## Build Commands

### Build Book PDF

```bash
cd doc/.src/book
bash make.sh nospell    # Skip spellcheck (faster)
bash make.sh            # With spellcheck
```

### Build Individual Chapter

```bash
cd doc/.src/chapters/vib   # or wave, diffu, advec, nonlin, trunc, softeng2, formulas
bash make.sh
```

### Run Tests

```bash
pytest src/                              # Run all tests
pytest src/vib/vib.py -v                 # Run tests in specific file
python -m pytest --tb=short              # With short traceback
```

### Linting

```bash
ruff check src/                          # Check for linting errors
isort --check-only src/                  # Check import ordering
isort src/                               # Fix import ordering
pre-commit run --all-files               # Run all pre-commit hooks
pre-commit run --hook-stage manual       # Run auto-fix hooks
```

## Architecture

### Directory Structure

- `src/` - Python source code organized by chapter (vib, wave, diffu, nonlin, etc.)
- `doc/.src/book/` - Main book build (book.do.txt includes all chapters)
- `doc/.src/chapters/` - DocOnce source for each chapter:
  - `*.do.txt` - DocOnce markup files (main content)
  - `fig-*/` - Figures for the chapter
  - `exer-*/` - Exercise solutions and supporting code
  - `.dict4spell.txt` - Chapter-specific spelling dictionary

### DocOnce Document Format

The book uses [DocOnce](https://github.com/doconce/doconce), a markup language that compiles to LaTeX/PDF. Key syntax:

- `@@@CODE path/to/file.py fromto: start_pattern@end_pattern` - Include code snippet between patterns
- `!bc pycod` / `!ec` - Python code block
- `!bt` / `!et` - LaTeX math block
- `idx{term}` - Index entry
- `ref{label}` - Cross-reference
- `# #include "file.do.txt"` - Include another file

### Code Organization Pattern

Each chapter's Python code lives in `src/CHAPTER/` and is referenced by documentation in `doc/.src/chapters/CHAPTER/`. The `@@@CODE` directive pulls code snippets directly from source files into the documentation, keeping code and docs in sync.

## Pre-commit Hooks

Pre-commit hooks run automatically on commit:

- trailing-whitespace, end-of-file-fixer
- isort (import ordering check)
- ruff (linting check)
- typos (spell check)
- markdownlint-cli2 (markdown check)

## Key Dependencies

- **doconce** - Document generation from .do.txt files
- **numpy, scipy, matplotlib, sympy** - Scientific Python stack for examples
- **pdflatex** - LaTeX compilation (requires TeX Live installation)

## Build Output

- `doc/.src/book/book.pdf` - Generated book PDF
- `doc/pub/book/pdf/fdm-book.pdf` - Published copy of book PDF
