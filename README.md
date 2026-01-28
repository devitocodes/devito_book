# Finite Difference Computing with PDEs

[![CI](https://github.com/devitocodes/devito_book/actions/workflows/ci.yml/badge.svg)](https://github.com/devitocodes/devito_book/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/devitocodes/devito_book/branch/devito/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito_book)

A modern approach to learning finite difference methods for partial differential equations, featuring [Devito](https://www.devitoproject.org/) for high-performance PDE solvers.

Based on *Finite Difference Computing with Partial Differential Equations* by Hans Petter Langtangen and Svein Linge, this edition has been modernized with:

- **[Quarto](https://quarto.org/)** for document generation (replacing DocOnce)
- **[Devito](https://www.devitoproject.org/)** DSL for symbolic PDE specification and automatic code generation
- **Modern Python** practices with type hints, testing, and CI/CD

## What is Devito?

Devito is a domain-specific language (DSL) embedded in Python for solving PDEs using finite differences. Instead of manually implementing stencil operations, you write mathematical expressions symbolically and Devito generates optimized C code:

```python
from devito import Grid, TimeFunction, Eq, Operator

# Define computational grid
grid = Grid(shape=(101,), extent=(1.0,))

# Create field with time derivative capability
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

# Write the wave equation symbolically
eq = Eq(u.dt2, c**2 * u.dx2)

# Devito generates optimized C code automatically
op = Operator([eq])
op.apply(time_M=100, dt=0.001)
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/devitocodes/devito_book.git
cd devito_book

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[devito]"

# Build the book
quarto render --to pdf
```

## Prerequisites

### Python 3.10+

```bash
pip install -e ".[devito]"      # With Devito support
pip install -e ".[devito-petsc]" # With PETSc for implicit schemes
pip install -e ".[dev]"          # Development tools
```

### Quarto

Install [Quarto](https://quarto.org/docs/get-started/):

```bash
# macOS
brew install quarto

# Ubuntu/Debian - see https://quarto.org/docs/get-started/ for current version
# Download the .deb from https://github.com/quarto-dev/quarto-cli/releases/latest
sudo dpkg -i quarto-*.deb
```

### LaTeX (for PDF output)

```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-full
```

## Building the Book

```bash
quarto render --to pdf    # PDF only
quarto render             # All formats (HTML + PDF)
quarto preview            # Live preview with hot reload
```

Output: `_book/Finite-Difference-Computing-with-PDEs.pdf`

## Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v -m "not devito"    # Skip Devito tests
pytest tests/ -v -m devito          # Devito tests only
```

## Book Structure

### Main Chapters

1. **Introduction to Devito** - DSL concepts, Grid, Function, TimeFunction, Operator
2. **Wave Equations** - 1D/2D wave propagation, absorbing boundaries, sources
3. **Diffusion Equations** - Heat equation, stability analysis, 2D extension
4. **Advection Equations** - Upwind schemes, Lax-Wendroff, CFL condition
5. **Nonlinear Problems** - Operator splitting, Burgers' equation, Picard iteration

### Appendices

- Finite difference formulas and derivations
- Truncation error analysis
- Software engineering practices

## Directory Structure

```
devito_book/
├── src/                      # Python solvers and utilities
│   ├── wave/                # Wave equation solvers (Devito)
│   ├── diffu/               # Diffusion solvers (Devito)
│   ├── advec/               # Advection solvers (Devito)
│   ├── nonlin/              # Nonlinear solvers (Devito)
│   ├── operators.py         # FD operators with symbolic derivation
│   └── verification.py      # Symbolic verification utilities
├── tests/                    # Pytest test suite
├── chapters/                 # Quarto book chapters
│   ├── devito_intro/        # Introduction to Devito
│   ├── wave/                # Wave equations
│   ├── diffu/               # Diffusion equations
│   ├── advec/               # Advection equations
│   ├── nonlin/              # Nonlinear problems
│   └── appendices/          # Reference material
├── _quarto.yml              # Book configuration
└── pyproject.toml           # Python package configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Make changes and run tests: `pytest tests/ -v`
6. Verify the build: `quarto render`
7. Submit a Pull Request

### Code Style

Pre-commit hooks enforce:
- **ruff** - Linting
- **isort** - Import sorting
- **typos** - Spell checking
- **markdownlint** - Markdown formatting

## License

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This work is adapted from:

> Langtangen, H.P., Linge, S. (2017). *Finite Difference Computing with PDEs: A Modern Software Approach*. Springer, Cham. [DOI: 10.1007/978-3-319-55456-3](https://doi.org/10.1007/978-3-319-55456-3)

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Authors

**Original Authors:**

- Hans Petter Langtangen (1962-2016)
- Svein Linge

**Adapted by:**

- Gerard J. Gorman, Imperial College London

## Links

- [Devito Project](https://www.devitoproject.org/)
- [Devito API Reference](https://www.devitoproject.org/api/)
- [Devito GitHub](https://github.com/devitocodes/devito)
- [Original Book (PDF)](https://hplgit.github.io/fdm-book/doc/pub/book/pdf/fdm-book-4print.pdf)
