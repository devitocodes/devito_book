# Finite Difference Computing with PDEs

[![Build Book PDF](https://github.com/devitocodes/devito_book/actions/workflows/build.yml/badge.svg)](https://github.com/devitocodes/devito_book/actions/workflows/build.yml)
[![CI](https://github.com/devitocodes/devito_book/actions/workflows/ci.yml/badge.svg)](https://github.com/devitocodes/devito_book/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/devitocodes/devito_book/branch/devito/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito_book)

Resources for the book *Finite Difference Computing with Partial Differential Equations - A Modern Software Approach* by Hans Petter Langtangen and Svein Linge.

> This easy-to-read book introduces the basics of solving partial differential
> equations by finite difference methods. The emphasis is on constructing
> finite difference schemes, formulating algorithms, implementing
> algorithms, verifying implementations, analyzing the physical behavior
> of the numerical solutions, and applying the methods and software
> to solve problems from physics and biology.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/devitocodes/devito_book.git
cd devito_book

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Build the book (using Quarto)
quarto render --to pdf
```

### With Devito (for PDE solvers)

```bash
# Install with Devito support (explicit schemes)
pip install -e ".[devito]"

# Or with PETSc branch for implicit schemes
pip install -e ".[devito-petsc]"
```

### Run Tests

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ -v -m "not devito"    # Skip Devito tests
```

## Prerequisites

### Python (3.10+)

The book uses [Quarto](https://quarto.org/) for document generation. Install Python dependencies with:

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (editable mode)
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Quarto

Install [Quarto](https://quarto.org/docs/get-started/) for document generation:

```bash
# macOS (Homebrew)
brew install quarto

# Ubuntu/Debian
wget https://github.com/quarto-dev/quarto-cli/releases/latest/download/quarto-linux-amd64.deb
sudo dpkg -i quarto-linux-amd64.deb
```

### LaTeX (TeX Live)

A full TeX Live installation is recommended. The book requires many LaTeX packages.

#### macOS

```bash
# Install BasicTeX (minimal) or full MacTeX
brew install --cask mactex  # Full installation (~4GB)
# OR
brew install --cask basictex  # Minimal (~100MB, then install packages below)

# If using BasicTeX, install required packages:
sudo tlmgr update --self
sudo tlmgr install \
    collection-latexrecommended \
    collection-fontsrecommended \
    collection-latexextra \
    collection-mathscience \
    mdframed needspace tcolorbox environ trimspaces \
    listings fancyvrb moreverb \
    microtype setspace relsize \
    titlesec appendix \
    caption subfig wrapfig \
    booktabs longtable ltablex tabularx multirow \
    hyperref bookmark \
    xcolor colortbl \
    tikz-cd pgf \
    bm soul \
    footmisc idxlayout tocbibind \
    chngcntr placeins \
    listingsutf8 \
    marvosym textcomp \
    cm-super lm
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-pictures \
    texlive-bibtex-extra \
    biber \
    latexmk \
    cm-super \
    dvipng \
    ghostscript
```

#### Windows

Install [MiKTeX](https://miktex.org/download) or [TeX Live for Windows](https://tug.org/texlive/windows.html). MiKTeX can automatically install missing packages on first use.

## Building the Book

### Full Book PDF

```bash
quarto render --to pdf
```

The PDF will be generated as `_book/Finite-Difference-Computing-with-PDEs.pdf`.

### All Formats (HTML + PDF)

```bash
quarto render
```

### Live Preview

```bash
quarto preview  # Opens browser with hot reload
```

## Directory Structure

```text
devito_book/
├── src/                      # Python package for book examples
│   ├── __init__.py          # Package exports
│   ├── symbols.py           # Canonical SymPy symbols
│   ├── operators.py         # Finite difference operators
│   ├── display.py           # LaTeX equation display utilities
│   ├── verification.py      # Symbolic verification utilities
│   ├── plotting.py          # Reproducible plotting
│   ├── common/              # Shared utilities
│   └── wave/                # Wave equation solvers
│       └── wave1D_devito.py # 1D wave solver using Devito
├── tests/                    # Pytest test suite
│   ├── conftest.py          # Test fixtures
│   ├── test_operators.py    # FD operator tests
│   ├── test_derivations.py  # Mathematical derivation tests
│   └── test_wave_devito.py  # Devito wave solver tests
├── chapters/                 # Quarto book chapters
│   ├── vib/                 # Vibration ODEs
│   ├── wave/                # Wave equations
│   ├── diffu/               # Diffusion equations
│   ├── advec/               # Advection equations
│   └── nonlin/              # Nonlinear problems
├── _quarto.yml              # Quarto book configuration
├── pyproject.toml           # Python package configuration
└── README.md                # This file
```

## Troubleshooting

### Missing LaTeX Packages

If pdflatex fails with "File not found" errors:

```bash
# Find the missing package
tlmgr search --global --file "missing-file.sty"

# Install it
sudo tlmgr install package-name
```

### Quarto Errors

Check the Quarto log output for details. For more verbose output:

```bash
quarto render --log-level debug
```

### Devito Installation Issues

If Devito installation fails, ensure you have a C compiler and dependencies:

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential python3-dev
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes
6. Run tests: `pytest tests/ -v`
7. Verify the build: `quarto render`
8. Commit your changes (pre-commit hooks run automatically)
9. Push to the branch and create a Pull Request

### Code Style

The project uses:
- **ruff** for linting
- **isort** for import sorting
- **typos** for spell checking
- **markdownlint** for markdown files

Pre-commit hooks enforce these automatically.

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Authors

- Hans Petter Langtangen (1962-2016)
- Svein Linge

## Acknowledgments

This book is part of a series on computational science. See also:

- [A Primer on Scientific Programming with Python](https://github.com/hplgit/primer)
- [Scaling of Differential Equations](https://github.com/hplgit/scaling-book)
