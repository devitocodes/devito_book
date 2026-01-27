# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source repository for *Finite Difference Computing with PDEs - A Modern Software Approach* by Hans Petter Langtangen and Svein Linge. The book teaches finite difference methods for solving PDEs through Python implementations.

## Build Commands

### Build Book PDF

```bash
quarto render --to pdf     # Build PDF
quarto render              # Build all formats (HTML + PDF)
quarto preview             # Live preview with hot reload
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

- `chapters/` - Quarto source files organized by topic:
  - `vib/` - Vibration ODEs
  - `wave/` - Wave equations
  - `diffu/` - Diffusion equations
  - `advec/` - Advection equations
  - `nonlin/` - Nonlinear problems
  - `appendices/` - Truncation errors, formulas, software engineering
- `src/` - Python source code organized by chapter
- `_book/` - Generated output (PDF, HTML)
- `_quarto.yml` - Book configuration

### Quarto Document Format

The book uses [Quarto](https://quarto.org/) for scientific publishing. Key syntax:

- ` ```python ` / ` ``` ` - Python code block
- `$$ ... $$` - Display math with optional `{#eq-label}`
- `@sec-label`, `@eq-label`, `@fig-label` - Cross-references
- `{{< include file.qmd >}}` - Include another file
- `{=latex}` blocks for raw LaTeX when needed

### Code Organization Pattern

Each chapter's Python code lives in `src/CHAPTER/` and can be included in QMD files using fenced code blocks or Quarto includes.

## Pre-commit Hooks

Pre-commit hooks run automatically on commit:

- trailing-whitespace, end-of-file-fixer
- isort (import ordering check)
- ruff (linting check)
- typos (spell check)
- markdownlint-cli2 (markdown check)

## Key Dependencies

- **quarto** - Document generation from .qmd files
- **numpy, scipy, matplotlib, sympy** - Scientific Python stack for examples
- **pdflatex** - LaTeX compilation (requires TeX Live installation)

## Build Output

- `_book/Finite-Difference-Computing-with-PDEs.pdf` - Generated book PDF

## Quarto Equation Labeling Guidelines

**Known Bug (GitHub Issue #2275)**: Quarto's `{#eq-label}` syntax cannot label individual lines within `\begin{align}` environments. This causes "macro parameter character #" LaTeX errors.

### What Fails

```markdown
$$
\begin{align}
a &= 0+1 {#eq-first}    <!-- causes LaTeX error -->
b &= 2+3 {#eq-second}
\end{align}
$$
```

### Working Patterns

**Single equation or whole block label** - place label AFTER closing `$$`:
```markdown
$$
\begin{split}
a &= 0+1 \\
b &= 2+3
\end{split}
$$ {#eq-block}
```

**Multiple separate equations** - use separate `$$` blocks:
```markdown
$$
a = 0+1
$$ {#eq-first}
$$
b = 2+3
$$ {#eq-second}
```

**Individual line labels in align** - use pure AMS LaTeX syntax:
```latex
\begin{align}
a &= 0+1 \label{eq:first} \\
b &= 2+3 \label{eq:second}
\end{align}

See Equation \eqref{eq:first} for details.
```

### Best Practices

- **Never mix Quarto `{#eq-}` and AMS `\label{}` syntax** in the same equation
- Use `\begin{equation}...\end{equation}` for single numbered equations
- Use `\begin{align}...\end{align}` with `\label{}` for multiple aligned, individually-numbered equations
- Use `\begin{aligned}...\end{aligned}` inside `\begin{equation}` for aligned equations sharing one number
- Add `*` (e.g., `\begin{align*}`) to suppress all numbering
- Reference with `\eqref{label}` for parenthesized numbers, `\ref{label}` for plain numbers
- For Quarto cross-refs, use `@eq-label` syntax with label placed after `$$`

### Cross-Reference Prefixes

| Type | Prefix | Example |
|------|--------|---------|
| Section | `@sec-` | `@sec-vib-ode1` |
| Equation | `@eq-` | `@eq-vib-ode1-step4` |
| Figure | `@fig-` | `@fig-vib-phase` |
| Table | `@tbl-` | `@tbl-trunc-fd1` |

Reference: [NMFS-OpenSci Quarto-AMS Math Guide](https://nmfs-opensci.github.io/quarto-amsmath)
