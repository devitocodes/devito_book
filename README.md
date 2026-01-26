# Finite Difference Computing with PDEs

[![Build Book PDF](https://github.com/devitocodes/devito_book/actions/workflows/build.yml/badge.svg)](https://github.com/devitocodes/devito_book/actions/workflows/build.yml)

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

# Build the book PDF
cd doc/.src/book
bash make.sh nospell
```

## Prerequisites

### Python (3.10+)

The book uses [DocOnce](https://github.com/doconce/doconce) for document generation. Install Python dependencies with:

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (editable mode)
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
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
cd doc/.src/book
bash make.sh nospell  # Skip spellcheck for faster builds
```

The PDF will be generated as `doc/.src/book/book.pdf`.

### With Spellcheck

```bash
cd doc/.src/book
bash make.sh  # Runs spellcheck before building
```

### Individual Chapters

Each chapter can be built independently:

```bash
cd doc/.src/chapters/vib  # or wave, diffu, etc.
bash make.sh
```

## Directory Structure

```text
devito_book/
├── src/                      # Source code for book examples
│   └── X/                    # Source code from chapter X
├── doc/
│   ├── pub/                  # Published documents
│   │   ├── book/            # Complete published book
│   │   └── X/               # Published chapter X
│   └── .src/                # DocOnce source
│       ├── book/            # Source for complete book
│       │   ├── make.sh      # Build script
│       │   ├── book.do.txt  # Main DocOnce file
│       │   └── preface.do.txt
│       └── chapters/
│           └── X/           # Source for chapter X
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Python dependencies (alternative)
└── README.md               # This file
```

## Troubleshooting

### DocOnce Configuration Errors

If you see permission errors related to DocOnce config:

```bash
export HOME=$(mktemp -d)  # Use temporary home directory
bash make.sh nospell
```

### Missing LaTeX Packages

If pdflatex fails with "File not found" errors:

```bash
# Find the missing package
tlmgr search --global --file "missing-file.sty"

# Install it
sudo tlmgr install package-name
```

### Encoding Errors

The build script automatically fixes encoding issues by converting from `utf8x` to standard `utf8`. If you see encoding errors, ensure you're using the latest `make.sh`.

### Build Logs

Check these files for detailed error information:

- `book.log` - LaTeX compilation log
- `book.dlog` - DocOnce processing log

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run the build to verify (`bash make.sh nospell`)
5. Commit your changes (`git commit -am 'Add improvement'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Authors

- Hans Petter Langtangen (1962-2016)
- Svein Linge

## Acknowledgments

This book is part of a series on computational science. See also:

- [A Primer on Scientific Programming with Python](https://github.com/hplgit/primer)
- [Scaling of Differential Equations](https://github.com/hplgit/scaling-book)
