# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source repository for *Finite Difference Computing with PDEs - A Modern Software Approach* by Hans Petter Langtangen and Svein Linge. The book teaches finite difference methods for solving PDEs using [Devito](https://www.devitoproject.org/), a domain-specific language for symbolic PDE specification.

### Key Technologies

- **Devito** - DSL for symbolic PDE specification and automatic code generation
- **Quarto** - Scientific publishing system for HTML and PDF output
- **SymPy** - Symbolic mathematics for derivations and verification

## Build Commands

### Build Book PDF

```bash
quarto render --to pdf     # Build PDF
quarto render              # Build all formats (HTML + PDF)
quarto preview             # Live preview with hot reload
```

### Run Tests

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ -v -m "not devito"    # Skip Devito tests
pytest tests/ -v -m devito          # Devito tests only
```

### Linting

```bash
ruff check src/                          # Check for linting errors
isort --check-only src/                  # Check import ordering
pre-commit run --all-files               # Run all pre-commit hooks
```

## Architecture

### Directory Structure

- `chapters/` - Quarto source files organized by topic:
  - `devito_intro/` - Introduction to Devito DSL
  - `wave/` - Wave equations (includes Devito solvers)
  - `diffu/` - Diffusion equations (includes Devito solvers)
  - `advec/` - Advection equations (includes Devito solvers)
  - `nonlin/` - Nonlinear problems (includes Devito solvers)
  - `appendices/` - Truncation errors, formulas, software engineering
- `src/` - Python source code:
  - `wave/` - Wave equation solvers (wave1D_devito.py, wave2D_devito.py)
  - `diffu/` - Diffusion solvers (diffu1D_devito.py, diffu2D_devito.py)
  - `advec/` - Advection solvers (advec1D_devito.py)
  - `nonlin/` - Nonlinear solvers (nonlin1D_devito.py)
  - `operators.py` - FD operators with symbolic derivation
  - `verification.py` - Symbolic verification utilities
- `tests/` - Pytest test suite
- `_book/` - Generated output (PDF, HTML)
- `_quarto.yml` - Book configuration

### Devito Patterns

When writing Devito code, follow these patterns:

```python
from devito import Grid, TimeFunction, Eq, Operator

# 1. Create a grid
grid = Grid(shape=(nx,), extent=(L,))

# 2. Create fields
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

# 3. Write equations symbolically
eq = Eq(u.forward, 2*u - u.backward + (c*dt)**2 * u.dx2)

# 4. Create and apply operator
op = Operator([eq])
op.apply(time_M=nt, dt=dt)
```

### Quarto Document Format

The book uses [Quarto](https://quarto.org/) for scientific publishing. Key syntax:

- ` ```python ` / ` ``` ` - Python code block
- `$$ ... $$` - Display math with optional `{#eq-label}`
- `@sec-label`, `@eq-label`, `@fig-label` - Cross-references
- `{{< include file.qmd >}}` - Include another file
- `{=latex}` blocks for raw LaTeX when needed

## Pre-commit Hooks

Pre-commit hooks run automatically on commit:

- trailing-whitespace, end-of-file-fixer
- isort (import ordering check)
- ruff (linting check)
- typos (spell check)
- markdownlint-cli2 (markdown check)

## Key Dependencies

- **devito** - PDE solver DSL (optional, for running solvers)
- **quarto** - Document generation from .qmd files
- **numpy, scipy, matplotlib, sympy** - Scientific Python stack
- **pdflatex** - LaTeX compilation (requires TeX Live)

## Build Output

- `_book/Finite-Difference-Computing-with-PDEs.pdf` - Generated book PDF

## Quarto Equation Labeling Guidelines

**Known Bug (GitHub Issue #2275)**: Quarto's `{#eq-label}` syntax cannot label individual lines within `\begin{align}` environments.

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

**Individual line labels in align** - use pure AMS LaTeX syntax:
```latex
\begin{align}
a &= 0+1 \label{eq:first} \\
b &= 2+3 \label{eq:second}
\end{align}
```

### Cross-Reference Prefixes

| Type | Prefix | Example |
|------|--------|---------|
| Section | `@sec-` | `@sec-vib-ode1` |
| Equation | `@eq-` | `@eq-vib-ode1-step4` |
| Figure | `@fig-` | `@fig-vib-phase` |
| Table | `@tbl-` | `@tbl-trunc-fd1` |
