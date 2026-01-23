# Build Status

## Current status: WORKING

The LaTeX build now completes successfully, producing a 730-page PDF.

**Build command:** `cd doc/.src/book && bash make.sh nospell`

## Summary of All Changes

### New Files Added

| File | Purpose |
|------|---------|
| `pyproject.toml` | Modern Python packaging (replaces setup.py). Use `pip install -e .` |
| `.github/workflows/build.yml` | GitHub Actions CI to build PDF on every push |
| `README.md` | Updated with comprehensive build instructions |
| `requirements.txt` | Simplified, references pyproject.toml |

### Source File Fixes

| File | Change |
|------|--------|
| `doc/.src/book/book.do.txt` | Removed `# Externaldocuments:` line (referenced non-existent files) |
| `doc/.src/chapters/mako_code.txt` | Fixed Python 3 syntax (`print` function) |
| `doc/.src/chapters/papers.bib` | Minor BibTeX field reordering (cosmetic) |

### Build Script Fixes (`doc/.src/book/make.sh`)

| Fix | Description |
|-----|-------------|
| Spellcheck conditional | Moved spellcheck commands inside the `nospell` conditional block |
| Shell escaping | Escaped `\colorlet` to `\\colorlet` in `edit_solution_admons` |
| Title page fix | Added post-processing to replace title block (avoids nested center environments) |
| **Encoding fix** | Changes `utf8x` to standard `utf8`, disables problematic `ucs` package |

## Root Cause Analysis

### The Encoding Error

Original error:
```
! Argument of Â has an extra }.
l.462 \clearpage
```

**Problem:** DocOnce generates LaTeX with deprecated `utf8x` input encoding:
```latex
\usepackage{ucs}
\usepackage[utf8x]{inputenc}
```

The `utf8x`/`ucs` combination causes issues with certain character/package combinations. The `Â` in the error is a `0xC2` byte being misinterpreted.

**Solution:** The `make.sh` post-processing script now converts to standard UTF-8:
```latex
%\usepackage{ucs}  % DISABLED
\usepackage[utf8]{inputenc}  % Changed from utf8x
```

### Note on `newcommands_keep.tex`

This file defines custom macros (e.g., `\u` for bold vectors). It must remain **enabled** - commenting it out causes undefined command errors in math environments.

## Dependencies

### Python

Install with: `pip install -e .` (uses pyproject.toml)

Core packages:
- `doconce>=1.5.15` - Document processor
- `mako>=1.3.0` - Template engine
- `pygments>=2.17.0` - Syntax highlighting
- `sphinx>=7.0.0` - Documentation

### LaTeX (TeX Live)

**macOS (BasicTeX + packages):**
```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install \
    collection-latexrecommended \
    collection-fontsrecommended \
    collection-latexextra \
    collection-mathscience \
    mdframed needspace listings fancyvrb \
    microtype hyperref bookmark \
    footmisc idxlayout tocbibind \
    cm-super lm
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    cm-super
```

## CI/CD

GitHub Actions workflow (`.github/workflows/build.yml`) runs on every push:
1. Sets up Python 3.12
2. Installs TeX Live packages
3. Installs Python dependencies via `pip install -e .`
4. Builds the PDF with `make.sh nospell`
5. Uploads PDF as artifact

## Quick Build Instructions

```bash
# Setup (one time)
python -m venv venv
source venv/bin/activate
pip install -e .

# Build
cd doc/.src/book
bash make.sh nospell

# Output: book.pdf (730 pages)
```
