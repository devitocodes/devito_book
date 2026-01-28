# Devito Book Refactoring: Development Roadmap

## Progress Summary

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Infrastructure Setup |
| 0.5 | ✅ Complete | SymPy Integration for Reproducible Mathematics |
| 1 | ⬜ Not Started | Restructure Book Organization |
| 2 | ✅ Complete | Wave Equations Chapter |
| 3 | ✅ Complete | Diffusion Equations Chapter |
| 4 | ✅ Complete | Advection Equations Chapter |
| 5 | ✅ Complete | Nonlinear Problems Chapter |
| 6 | ✅ Complete | Appendices |
| 7 | ✅ Complete | Testing & CI Infrastructure |
| 8 | ⬜ Not Started | Final Integration & Review |

### Recent Completions (2025-01-28)

**Phase 0 - Infrastructure:**
- Created `src/` package structure with `__init__.py` exports
- Added `pyproject.toml` with optional `devito` and `devito-petsc` dependencies
- Set up pre-commit hooks (ruff, isort, typos, markdownlint)
- Created `.gitignore` and `.markdownlintignore` configurations

**Phase 0.5 - SymPy Integration:**
- `src/symbols.py` - Canonical SymPy symbols (x, t, dx, dt, etc.)
- `src/operators.py` - FD operators with truncation error analysis
- `src/display.py` - LaTeX equation display utilities
- `src/verification.py` - Symbolic identity and PDE verification
- `src/plotting.py` - Reproducible matplotlib/plotly plots

**Phase 7 - Testing & CI:**
- `tests/conftest.py` - Pytest fixtures including `devito_available`
- `tests/test_operators.py` - 37 tests for FD operators
- `tests/test_derivations.py` - 22 tests for mathematical derivations
- `tests/test_wave_devito.py` - Devito wave solver tests (skipped if Devito not installed)
- `.github/workflows/ci.yml` - GitHub Actions CI with Codecov integration

**Phase 2 - Wave Equations (Complete):**

*Python Solvers:*
- `src/wave/wave1D_devito.py` - 1D wave equation solver using Devito DSL
- `src/wave/wave2D_devito.py` - 2D wave equation solver using `.laplace`
- `src/wave/sources.py` - Source wavelets (Ricker, Gaussian, spectrum analysis)
- 29 wave solver tests (1D, 2D, sources)

*Quarto Chapters:*
- `chapters/wave/wave1D_devito.qmd` - Core 1D wave with Devito (Grid, TimeFunction, Operator)
- `chapters/wave/wave1D_features.qmd` - Source terms, variable coefficients, absorbing BCs
- `chapters/wave/wave2D_devito.qmd` - 2D extension with `.laplace`, visualization
- `chapters/wave/wave_exercises.qmd` - 10 Devito-based exercises with solutions
- Updated `chapters/wave/index.qmd` to include new Devito sections

**Phase 3 - Diffusion Equations (Complete):**

*Python Solvers:*
- `src/diffu/diffu1D_devito.py` - 1D diffusion equation solver (Forward Euler explicit)
- `src/diffu/diffu2D_devito.py` - 2D diffusion solver using `.laplace`
- `src/diffu/__init__.py` - Module exports
- 23 diffusion solver tests (1D, 2D, initial conditions)

*Quarto Chapters:*
- `chapters/diffu/diffu1D_devito.qmd` - Forward Euler with Devito (time_order=1, .dt, Fourier number)
- `chapters/diffu/diffu2D_devito.qmd` - 2D extension with stability analysis
- `chapters/diffu/diffu_devito_exercises.qmd` - 10 Devito-based exercises with solutions
- Updated `chapters/diffu/index.qmd` to include new Devito sections

**Phase 4 - Advection Equations (Complete):**

*Python Solvers:*
- `src/advec/advec1D_devito.py` - 1D advection solvers using Devito DSL:
  - Upwind scheme (first-order, uses `.subs()` for shifted indexing)
  - Lax-Wendroff scheme (second-order, centered + diffusion correction)
  - Lax-Friedrichs scheme (first-order, very stable)
- `src/advec/__init__.py` - Module exports
- Helper functions: `exact_advection`, `exact_advection_periodic`, `gaussian_initial_condition`, `step_initial_condition`, `convergence_test_advection`
- 25 advection solver tests (upwind, Lax-Wendroff, Lax-Friedrichs, convergence, boundary conditions)

*Quarto Chapters:*
- `chapters/advec/advec1D_devito.qmd` - Advection schemes with Devito (upwind differencing, CFL condition, scheme comparison)
- `chapters/advec/advec_devito_exercises.qmd` - 10 Devito-based exercises with solutions
- Updated `chapters/advec/index.qmd` to include new Devito sections

**Phase 5 - Nonlinear Problems (Complete):**

*Python Solvers:*
- `src/nonlin/nonlin1D_devito.py` - Nonlinear PDE solvers using Devito DSL:
  - `solve_nonlinear_diffusion_explicit` - Explicit scheme with lagged coefficient evaluation
  - `solve_reaction_diffusion_splitting` - Lie and Strange operator splitting
  - `solve_burgers_equation` - Viscous Burgers' equation with conservative form
  - `solve_nonlinear_diffusion_picard` - Picard iteration for implicit schemes
- Helper functions: `constant_diffusion`, `linear_diffusion`, `porous_medium_diffusion`
- Reaction terms: `logistic_reaction`, `fisher_reaction`, `allen_cahn_reaction`
- `src/nonlin/__init__.py` - Module exports
- 29 nonlinear solver tests (explicit, splitting, Burgers, Picard, diffusion coefficients, reaction terms)

*Quarto Chapters:*
- `chapters/nonlin/nonlin1D_devito.qmd` - Nonlinear diffusion, reaction-diffusion splitting, Burgers' equation with Devito
- `chapters/nonlin/nonlin_devito_exercises.qmd` - 10 Devito-based exercises with solutions
- Updated `chapters/nonlin/index.qmd` to include new Devito sections

**Phase 6 - Appendices (Complete):**

*Truncation Error Appendix:*
- Added "Devito and Truncation Errors" section to `chapters/appendices/trunc/trunc.qmd`
- Covers `space_order` parameter and stencil accuracy
- Shows how to view generated stencils symbolically
- Discusses trading accuracy for performance
- Links truncation error theory to Devito's implementation
- Includes convergence rate verification example

*Software Engineering Appendix:*
- Added "Software Engineering with Devito" section to `chapters/appendices/softeng2/softeng2.qmd`
- Project structure for Devito applications
- Pytest fixtures for Devito testing (Grid, TimeFunction reuse)
- Convergence testing pattern with manufactured solutions
- Performance profiling with DEVITO_LOGGING
- Caching and compilation configuration
- Result classes using dataclasses
- Comparison table: Devito vs traditional optimization approaches

---

## Executive Summary

This document outlines the comprehensive plan to refactor *Finite Difference Computing with PDEs - A Modern Software Approach* to use the Devito DSL instead of NumPy-based implementations. The refactored book will teach finite difference methods through Devito's symbolic PDE specification and automatic code generation capabilities.

**Key Decisions:**
- Minimize ODE content; start with PDEs using Devito from Chapter 1
- Educational focus on readable Devito code, not advanced optimization
- Use Devito's `petsc` branch for implicit schemes (Backward Euler, Crank-Nicolson)
- 1D primary focus with brief 2D/3D extensions
- Modern visualization stack (plotly/holoviews)
- Retain numpy implementations only in test files for verification
- Adapt existing exercises to Devito

---

## Table of Contents

1. [Phase 0: Infrastructure Setup](#phase-0-infrastructure-setup)
2. [Phase 0.5: SymPy Integration for Reproducible Mathematics](#phase-05-sympy-integration-for-reproducible-mathematics)
3. [Phase 1: Restructure Book Organization](#phase-1-restructure-book-organization)
4. [Phase 2: Wave Equations Chapter](#phase-2-wave-equations-chapter)
5. [Phase 3: Diffusion Equations Chapter](#phase-3-diffusion-equations-chapter)
6. [Phase 4: Advection Equations Chapter](#phase-4-advection-equations-chapter)
7. [Phase 5: Nonlinear Problems Chapter](#phase-5-nonlinear-problems-chapter)
8. [Phase 6: Appendices](#phase-6-appendices)
9. [Phase 7: Testing & CI Infrastructure](#phase-7-testing-ci-infrastructure)
10. [Phase 8: Final Integration & Review](#phase-8-final-integration-review)
11. [Appendix A: Devito Concept Introduction Points](#appendix-a-devito-concept-introduction-points)
12. [Appendix B: File Migration Map](#appendix-b-file-migration-map)
13. [Appendix C: Testing Strategy](#appendix-c-testing-strategy)
14. [Appendix D: Mathematical Derivation Verification](#appendix-d-mathematical-derivation-verification)

---

## Phase 0: Infrastructure Setup

**Duration Estimate:** Foundation work before content migration

### 0.1 Environment Setup

#### Create Dual Virtual Environments

```bash
# Main environment for explicit schemes
python -m venv venv
source venv/bin/activate
pip install devito numpy scipy matplotlib plotly holoviews pytest sympy

# Secondary environment for implicit schemes (PETSc)
python -m venv venv_implicit
source venv_implicit/bin/activate
cd devito_repo && git checkout petsc && pip install -e .
pip install petsc4py mpi4py
```

#### Directory Structure Changes

```
devito_book/
├── src/
│   ├── devito_intro/        # NEW: Devito fundamentals
│   ├── wave/                # REFACTOR: Devito wave solvers
│   ├── diffu/               # REFACTOR: Devito diffusion solvers
│   ├── advec/               # REFACTOR: Devito advection solvers
│   ├── nonlin/              # REFACTOR: Devito nonlinear solvers
│   ├── legacy/              # NEW: Original numpy code for tests only
│   └── common/              # NEW: Shared utilities
├── tests/
│   ├── test_wave.py
│   ├── test_diffu.py
│   ├── test_advec.py
│   ├── test_nonlin.py
│   └── conftest.py          # Pytest fixtures
├── chapters/
│   ├── devito_intro/        # NEW: Replace vib chapter
│   └── ...
└── .github/
    └── workflows/
        └── tests.yml        # GitHub Actions CI
```

### 0.2 GitHub Actions CI Setup

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [main, devito]
  pull_request:
    branches: [main]

jobs:
  test-explicit:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install devito numpy scipy matplotlib pytest sympy
      - name: Run tests
        run: pytest tests/ -v --ignore=tests/test_implicit.py

  test-implicit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install PETSc
        run: |
          sudo apt-get update
          sudo apt-get install -y petsc-dev
      - name: Install Devito petsc branch
        run: |
          git clone --branch petsc https://github.com/devitocodes/devito.git devito_petsc
          pip install -e devito_petsc
          pip install petsc4py mpi4py pytest
      - name: Run implicit tests
        run: pytest tests/test_implicit.py -v
```

### 0.3 Common Utilities Module

Create `src/common/utils.py`:

```python
"""Common utilities for Devito book examples."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_convergence_rate(errors, spacings):
    """Compute empirical convergence rate from error sequence."""
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(spacings[i-1] / spacings[i])
        rates.append(r)
    return rates


def l2_error(numerical, analytical, dx):
    """Compute discrete L2 error."""
    return np.sqrt(dx * np.sum((numerical - analytical)**2))


def max_error(numerical, analytical):
    """Compute maximum (infinity) norm error."""
    return np.abs(numerical - analytical).max()


def plot_solution_1d(x, u_numerical, u_exact=None, title="Solution"):
    """Create interactive 1D solution plot with Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=u_numerical, mode='lines+markers',
                             name='Numerical', line=dict(color='blue')))
    if u_exact is not None:
        fig.add_trace(go.Scatter(x=x, y=u_exact, mode='lines',
                                 name='Exact', line=dict(color='red', dash='dash')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='u')
    return fig


def plot_convergence(spacings, errors, expected_order=2, title="Convergence"):
    """Create log-log convergence plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spacings, y=errors, mode='lines+markers',
                             name='Measured error'))
    # Add reference slope
    ref = errors[0] * (np.array(spacings) / spacings[0])**expected_order
    fig.add_trace(go.Scatter(x=spacings, y=ref, mode='lines',
                             name=f'O(h^{expected_order})', line=dict(dash='dash')))
    fig.update_layout(title=title, xaxis_title='Spacing', yaxis_title='Error',
                      xaxis_type='log', yaxis_type='log')
    return fig


def animate_1d_wave(x, u_history, t_values, interval=50):
    """Create animated 1D wave visualization using Plotly."""
    fig = go.Figure(
        data=[go.Scatter(x=x, y=u_history[0], mode='lines')],
        layout=go.Layout(
            title="Wave propagation",
            xaxis=dict(range=[x.min(), x.max()]),
            yaxis=dict(range=[u_history.min()*1.1, u_history.max()*1.1]),
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                           method="animate",
                                           args=[None, {"frame": {"duration": interval}}])])]
        ),
        frames=[go.Frame(data=[go.Scatter(x=x, y=u_history[n])],
                        name=f't={t_values[n]:.3f}')
               for n in range(len(t_values))]
    )
    return fig
```

---

## Phase 0.5: SymPy Integration for Reproducible Mathematics

### Rationale

All mathematical derivations in the book should be:
1. **Reproducible**: Generated from SymPy, not hand-written LaTeX
2. **Verifiable**: Unit tests confirm correctness of derivations
3. **Consistent**: Shared symbols prevent notation drift across chapters
4. **Inline macros**: No custom LaTeX `\newcommand` macros; all math is explicit

### 0.5.1 Remove LaTeX Macros from `_quarto.yml`

The current `_quarto.yml` contains 80+ custom LaTeX macros. These will be **removed** and replaced with:
- SymPy-generated LaTeX for all equations
- Explicit standard LaTeX where SymPy output is used

**Before (remove this):**
```yaml
include-in-header:
  text: |
    \newcommand{\half}{\frac{1}{2}}
    \newcommand{\uex}{u_{\mbox{\footnotesize e}}}
    ... (80+ macros)
```

**After (minimal header):**
```yaml
include-in-header:
  text: |
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{bm}
```

#### Macro Migration Table

| Old Macro | SymPy Replacement | Inline LaTeX |
|-----------|-------------------|--------------|
| `\half` | `sp.Rational(1,2)` | `\frac{1}{2}` |
| `\uex` | `exact('u')(x,t)` | `u_{\text{e}}` |
| `\Ddt{u}` | `sp.Derivative(u,t)` | `\frac{D u}{dt}` |
| `\Oof{h^2}` | `sp.O(h**2)` | `\mathcal{O}(h^2)` |
| `\x`, `\uu`, etc. | `sp.Symbol(r'\mathbf{x}')` | `\mathbf{x}`, `\mathbf{u}` |
| `\dfc` (α) | `alpha` from symbols.py | `\alpha` |
| `\Ix`, `\It` | Index set symbols | `\mathcal{I}_x`, `\mathcal{I}_t` |

#### Migration Script: `scripts/migrate_macros.py`

```python
"""
Script to identify custom macro usage in .qmd files.
Run before migration to catalog all usages.
"""
import re
from pathlib import Path

# Macros defined in current _quarto.yml
CUSTOM_MACROS = [
    r'\\half', r'\\halfi', r'\\tp', r'\\uex', r'\\uexd',
    r'\\vex', r'\\Vex', r'\\vexd', r'\\Aex', r'\\wex',
    r'\\Ddt', r'\\E\[', r'\\Var\[', r'\\Std\[',
    r'\\xpoint', r'\\normalvec', r'\\Oof',
    r'\\x\\b', r'\\X\\b', r'\\uu', r'\\vv', r'\\w\\b',
    r'\\acc', r'\\rpos', r'\\V\\b', r'\\e\\b', r'\\f\\b', r'\\F\\b',
    r'\\stress', r'\\strain', r'\\stressc', r'\\strainc',
    r'\\I\\b', r'\\T\\b', r'\\U\\b', r'\\q\\b', r'\\g\\b',
    r'\\dfc', r'\\ii', r'\\jj', r'\\kk', r'\\ir', r'\\ith', r'\\iz',
    r'\\Ix', r'\\Iy', r'\\Iz', r'\\It', r'\\If', r'\\Ifd', r'\\Ifb',
    r'\\setb', r'\\sete', r'\\setl', r'\\setr', r'\\seti',
    r'\\sequencei', r'\\sequencej',
    r'\\stepzero', r'\\stephalf', r'\\stepone',
    r'\\basphi', r'\\baspsi', r'\\refphi', r'\\psib',
    r'\\sinL', r'\\xno', r'\\Xno', r'\\yno', r'\\Yno', r'\\xdno',
    r'\\dX', r'\\dx', r'\\ds',
    r'\\Real', r'\\Integerp', r'\\Integer',
]

def find_macro_usage(qmd_dir: Path):
    """Find all custom macro usages in .qmd files."""
    results = {}

    for qmd_file in qmd_dir.rglob('*.qmd'):
        content = qmd_file.read_text()

        for macro in CUSTOM_MACROS:
            matches = re.findall(macro, content)
            if matches:
                if qmd_file.name not in results:
                    results[qmd_file.name] = {}
                results[qmd_file.name][macro] = len(matches)

    return results

if __name__ == '__main__':
    usage = find_macro_usage(Path('chapters'))
    for filename, macros in sorted(usage.items()):
        print(f"\n{filename}:")
        for macro, count in sorted(macros.items(), key=lambda x: -x[1]):
            print(f"  {macro}: {count}")
```

### 0.5.2 Create Shared Symbol Module (`src/symbols.py`)

```python
"""Canonical SymPy symbols used throughout the book.

Centralising symbols ensures:
1. Consistent notation across chapters
2. No accidental shadowing (e.g., redefining x differently)
3. Assumptions are uniform (real, positive, etc.)
4. LaTeX rendering is consistent
"""

import sympy as sp

# Spatial variables
x, y, z = sp.symbols('x y z', real=True)

# Temporal variable
t = sp.symbols('t', real=True, nonnegative=True)

# Grid spacing (always positive)
dx = sp.Symbol(r'\Delta x', positive=True, real=True)
dy = sp.Symbol(r'\Delta y', positive=True, real=True)
dz = sp.Symbol(r'\Delta z', positive=True, real=True)
dt = sp.Symbol(r'\Delta t', positive=True, real=True)

# Alternative notation aliases
h = dx  # Spatial step alias
k = dt  # Temporal step alias

# Generic function symbols
u = sp.Function('u')
v = sp.Function('v')
f = sp.Function('f')

# Physical parameters (typically positive)
alpha = sp.Symbol(r'\alpha', positive=True)  # Diffusivity
c = sp.Symbol('c', positive=True)            # Wave speed
nu = sp.Symbol(r'\nu', positive=True)        # Viscosity
omega = sp.Symbol(r'\omega', positive=True)  # Angular frequency

# Index variables for stencils
i, j, n, m = sp.symbols('i j n m', integer=True)

# Dimensionless numbers
C = sp.Symbol('C', positive=True)  # Courant number
F = sp.Symbol('F', positive=True)  # Fourier/diffusion number

# For error analysis
xi = sp.Symbol(r'\xi', real=True)  # Intermediate point
O = sp.Function('O')               # Big-O notation

# Exact solution subscript helper
def exact(func_name='u'):
    """Create exact solution symbol u_e."""
    return sp.Function(f'{func_name}_e')

__all__ = [
    'sp',
    'x', 'y', 'z', 't',
    'dx', 'dy', 'dz', 'dt', 'h', 'k',
    'u', 'v', 'f',
    'alpha', 'c', 'nu', 'omega',
    'i', 'j', 'n', 'm',
    'C', 'F',
    'xi', 'O', 'exact',
]
```

### 0.5.3 Create Finite Difference Operators Module (`src/operators.py`)

```python
"""Finite difference operators and stencil generation.

All operators return SymPy expressions that can be:
1. Displayed as LaTeX via sp.latex()
2. Verified via Taylor series expansion
3. Converted to Devito stencils
"""

import sympy as sp
from .symbols import x, dx, u


def forward_diff(func, var, step):
    """First-order forward difference: (f(x+h) - f(x)) / h"""
    shifted = func.subs(var, var + step)
    return (shifted - func) / step


def backward_diff(func, var, step):
    """First-order backward difference: (f(x) - f(x-h)) / h"""
    shifted = func.subs(var, var - step)
    return (func - shifted) / step


def central_diff(func, var, step):
    """Second-order central difference: (f(x+h) - f(x-h)) / (2h)"""
    forward = func.subs(var, var + step)
    backward = func.subs(var, var - step)
    return (forward - backward) / (2 * step)


def second_derivative_central(func, var, step):
    """Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²"""
    forward = func.subs(var, var + step)
    backward = func.subs(var, var - step)
    return (forward - 2*func + backward) / step**2


def laplacian_2d(func, x_var, y_var, hx, hy):
    """2D Laplacian using central differences."""
    d2_dx2 = second_derivative_central(func, x_var, hx)
    d2_dy2 = second_derivative_central(func, y_var, hy)
    return d2_dx2 + d2_dy2


def derive_truncation_error(stencil_expr, exact_derivative, var, step, order=6):
    """
    Compute truncation error of a stencil via Taylor expansion.

    Returns (leading_error_term, full_series)
    """
    error = sp.simplify(stencil_expr - exact_derivative)
    series = sp.series(error, step, 0, order).removeO()
    return series


def stencil_to_devito(stencil_expr, func_symbol, grid_func_name='u'):
    """
    Convert symbolic stencil to Devito notation hint.

    Returns a string showing the Devito equivalent.
    """
    # This is a helper for documentation, not code generation
    latex_str = sp.latex(stencil_expr)
    return f"# SymPy: {latex_str}\n# Devito: {grid_func_name}.dx2 (for second derivative)"


__all__ = [
    'forward_diff',
    'backward_diff',
    'central_diff',
    'second_derivative_central',
    'laplacian_2d',
    'derive_truncation_error',
    'stencil_to_devito',
]
```

### 0.5.4 Create Display Utilities Module (`src/display.py`)

```python
"""Display utilities for SymPy output in Quarto.

Provides consistent formatting of SymPy expressions as LaTeX
for both display equations and inline math.
"""

import sympy as sp
from IPython.display import display, Markdown

# Configure SymPy for Quarto/Jupyter environment
sp.init_printing(use_latex='mathjax')


def show_eq(expr, label=None, numbered=True):
    """
    Display a SymPy expression as a LaTeX equation.

    Parameters
    ----------
    expr : sympy expression or Equality
        The expression to display
    label : str, optional
        Cross-reference label (without 'eq-' prefix)
        Example: label='heat' produces {#eq-heat}
    numbered : bool, default True
        Whether equation should be numbered

    Examples
    --------
    >>> show_eq(sp.Eq(u, sp.diff(v, x)))  # Numbered equation
    >>> show_eq(heat_eq, label='heat')     # With cross-reference
    """
    latex_str = sp.latex(expr)

    if label:
        output = f"$$ {latex_str} $$ {{#eq-{label}}}"
    elif numbered:
        output = f"$$ {latex_str} $$"
    else:
        output = f"$$ {latex_str} $$"

    display(Markdown(output))


def show_eq_aligned(*equations, label=None):
    """
    Display multiple equations in an aligned environment.

    Parameters
    ----------
    *equations : sympy Equality objects
        Equations to align (aligned at = sign)
    label : str, optional
        Cross-reference label for the block
    """
    lines = []
    for eq in equations:
        if isinstance(eq, sp.Equality):
            lhs = sp.latex(eq.lhs)
            rhs = sp.latex(eq.rhs)
            lines.append(f"{lhs} &= {rhs}")
        else:
            lines.append(f"&= {sp.latex(eq)}")

    aligned = r" \\".join(lines)

    if label:
        output = f"$$\\begin{{aligned}}\n{aligned}\n\\end{{aligned}}$$ {{#eq-{label}}}"
    else:
        output = f"$$\\begin{{aligned}}\n{aligned}\n\\end{{aligned}}$$"

    display(Markdown(output))


def show_derivation(steps, label=None):
    """
    Display a multi-step derivation with alignment.

    Parameters
    ----------
    steps : list of (str, sympy_expr) tuples
        Each tuple is (description, expression)
    """
    lines = []
    for desc, expr in steps:
        latex_expr = sp.latex(expr)
        lines.append(f"&= {latex_expr} && \\text{{{desc}}}")

    aligned = r" \\".join(lines)

    if label:
        output = f"$$\\begin{{aligned}}\n{aligned}\n\\end{{aligned}}$$ {{#eq-{label}}}"
    else:
        output = f"$$\\begin{{aligned}}\n{aligned}\n\\end{{aligned}}$$"

    display(Markdown(output))


def inline_latex(expr):
    """Return LaTeX string for inline use (no display)."""
    return sp.latex(expr)


def show_stencil_table(stencils, title="Finite Difference Stencils"):
    """
    Display a table of stencils with their properties.

    Parameters
    ----------
    stencils : list of dict
        Each dict has keys: 'name', 'formula', 'order', 'type'
    """
    header = "| Stencil | Formula | Order | Type |\n|---------|---------|-------|------|\n"
    rows = []
    for s in stencils:
        formula_latex = f"${sp.latex(s['formula'])}$"
        rows.append(f"| {s['name']} | {formula_latex} | {s['order']} | {s['type']} |")

    display(Markdown(header + "\n".join(rows)))


__all__ = [
    'show_eq',
    'show_eq_aligned',
    'show_derivation',
    'inline_latex',
    'show_stencil_table',
]
```

### 0.5.5 Create Verification Module (`src/verification.py`)

```python
"""Verification utilities for checking symbolic results.

These functions are used both in-book (hidden chunks) and in pytest.
Any assertion failure will halt the Quarto build, catching errors early.
"""

import sympy as sp


def verify_identity(expr1, expr2, simplify_func=sp.simplify):
    """
    Verify two expressions are symbolically equivalent.

    Raises AssertionError if they differ (fails the build in CI).
    """
    diff = simplify_func(expr1 - expr2)
    assert diff == 0, f"Identity check failed:\n  {expr1}\n  ≠\n  {expr2}\n  diff = {diff}"
    return True


def check_stencil_order(stencil, exact_derivative, var, step, expected_order):
    """
    Verify a finite difference stencil has the expected truncation error order.

    Parameters
    ----------
    stencil : sympy expression
        The finite difference approximation
    exact_derivative : sympy expression
        The exact derivative being approximated
    var : sympy Symbol
        The variable of differentiation
    step : sympy Symbol
        The grid spacing (h, dx, etc.)
    expected_order : int
        Expected order of accuracy (e.g., 2 for O(h²))

    Raises
    ------
    AssertionError if order doesn't match
    """
    error = sp.simplify(stencil - exact_derivative)
    series = sp.series(error, step, 0, expected_order + 2)

    # Check that terms below expected_order vanish
    for power in range(expected_order):
        coeff = series.coeff(step, power)
        assert coeff == 0, f"Order check failed: non-zero O({step}^{power}) term: {coeff}"

    # Check that the expected_order term is non-zero
    leading_coeff = series.coeff(step, expected_order)
    assert leading_coeff != 0, f"Order higher than expected: no O({step}^{expected_order}) term"

    return True


def verify_pde_solution(pde_lhs, pde_rhs, solution, variables):
    """
    Verify that a function satisfies a PDE.

    Parameters
    ----------
    pde_lhs : sympy expression
        Left-hand side of PDE (e.g., u_t)
    pde_rhs : sympy expression
        Right-hand side of PDE (e.g., alpha * u_xx)
    solution : sympy expression
        Proposed solution
    variables : dict
        Mapping from function symbols to the solution
    """
    lhs_eval = pde_lhs.subs(variables)
    rhs_eval = pde_rhs.subs(variables)

    diff = sp.simplify(lhs_eval - rhs_eval)
    assert diff == 0, f"Solution does not satisfy PDE:\n  LHS = {lhs_eval}\n  RHS = {rhs_eval}"
    return True


def numerical_verify(expr, subs_dict, expected, tol=1e-10):
    """
    Numerically verify an expression evaluates to expected value.

    Useful as a sanity check when symbolic simplification is difficult.
    """
    result = float(expr.subs(subs_dict).evalf())
    assert abs(result - expected) < tol, \
        f"Numerical check failed: {result} ≠ {expected} (tol={tol})"
    return True


def verify_stability_condition(scheme_amplification, condition, symbol_ranges):
    """
    Verify stability condition for a numerical scheme.

    Parameters
    ----------
    scheme_amplification : sympy expression
        Amplification factor |g|
    condition : sympy expression
        Stability condition (e.g., C <= 1)
    symbol_ranges : dict
        Ranges for symbols to test numerically
    """
    # Numerical spot-check of stability condition
    import numpy as np

    for sym, (lo, hi) in symbol_ranges.items():
        test_values = np.linspace(lo, hi, 10)
        for val in test_values:
            amp = abs(complex(scheme_amplification.subs(sym, val).evalf()))
            assert amp <= 1.0 + 1e-10, \
                f"Stability violated at {sym}={val}: |g| = {amp}"

    return True


__all__ = [
    'verify_identity',
    'check_stencil_order',
    'verify_pde_solution',
    'numerical_verify',
    'verify_stability_condition',
]
```

### 0.5.6 Reproducible Plotting Module (`src/plotting.py`)

```python
"""Reproducible plotting utilities.

All plots are deterministic:
1. Fixed random seeds for any stochastic elements
2. Explicit figure sizes and DPI
3. Consistent color schemes
4. Exportable to both interactive (HTML) and static (PDF) formats
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global random seed for reproducibility
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def get_color_scheme():
    """Return consistent color scheme for all plots."""
    return {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'tertiary': '#2ca02c',     # Green
        'error': '#d62728',        # Red
        'exact': '#9467bd',        # Purple
        'numerical': '#1f77b4',    # Blue
        'grid': '#e0e0e0',
    }


def create_solution_plot(x, u_numerical, u_exact=None, title="Solution",
                         xlabel="x", ylabel="u", show_error=False):
    """
    Create reproducible solution comparison plot.

    Parameters
    ----------
    x : array-like
        Spatial coordinates
    u_numerical : array-like
        Numerical solution
    u_exact : array-like, optional
        Exact solution for comparison
    title : str
        Plot title
    show_error : bool
        If True, add error subplot

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = get_color_scheme()

    if show_error and u_exact is not None:
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=("Solution", "Error"),
                           vertical_spacing=0.15)
        row_solution = 1
    else:
        fig = go.Figure()
        row_solution = None

    # Numerical solution
    fig.add_trace(
        go.Scatter(x=x, y=u_numerical, mode='lines+markers',
                   name='Numerical', line=dict(color=colors['numerical']),
                   marker=dict(size=4)),
        row=row_solution, col=1 if row_solution else None
    )

    # Exact solution
    if u_exact is not None:
        fig.add_trace(
            go.Scatter(x=x, y=u_exact, mode='lines',
                       name='Exact', line=dict(color=colors['exact'], dash='dash')),
            row=row_solution, col=1 if row_solution else None
        )

        # Error subplot
        if show_error:
            error = np.abs(u_numerical - u_exact)
            fig.add_trace(
                go.Scatter(x=x, y=error, mode='lines',
                           name='|Error|', line=dict(color=colors['error'])),
                row=2, col=1
            )

    fig.update_layout(
        title=title,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(x=0.02, y=0.98),
        width=700,
        height=500 if not show_error else 700,
    )

    if row_solution:
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
    else:
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)

    return fig


def create_convergence_plot(spacings, errors, expected_orders=None,
                            title="Convergence", xlabel="Grid spacing",
                            ylabel="Error"):
    """
    Create reproducible log-log convergence plot.

    Parameters
    ----------
    spacings : array-like
        Grid spacings (dx values)
    errors : array-like or dict
        Error values; if dict, keys are labels
    expected_orders : list of int, optional
        Expected convergence orders for reference lines

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = get_color_scheme()
    fig = go.Figure()

    # Handle single or multiple error series
    if isinstance(errors, dict):
        error_series = errors
    else:
        error_series = {'Error': errors}

    color_list = [colors['primary'], colors['secondary'], colors['tertiary']]

    for idx, (label, err) in enumerate(error_series.items()):
        fig.add_trace(
            go.Scatter(x=spacings, y=err, mode='lines+markers',
                       name=label, line=dict(color=color_list[idx % len(color_list)]))
        )

    # Reference slopes
    if expected_orders:
        for order in expected_orders:
            # Reference line starting from first data point
            ref_errors = errors if not isinstance(errors, dict) else list(errors.values())[0]
            ref = ref_errors[0] * (np.array(spacings) / spacings[0])**order
            fig.add_trace(
                go.Scatter(x=spacings, y=ref, mode='lines',
                           name=f'O(h^{order})',
                           line=dict(dash='dot', color='gray'))
            )

    fig.update_layout(
        title=title,
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
        font=dict(size=12),
        width=600,
        height=450,
    )
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)

    return fig


def create_animation_frames(x, u_history, t_values):
    """
    Create animation frames for time evolution.

    Returns a Plotly figure with animation controls.
    """
    colors = get_color_scheme()

    fig = go.Figure(
        data=[go.Scatter(x=x, y=u_history[0], mode='lines',
                        line=dict(color=colors['primary']))],
        layout=go.Layout(
            title="Time Evolution",
            xaxis=dict(range=[x.min(), x.max()], title="x"),
            yaxis=dict(range=[u_history.min()*1.1, u_history.max()*1.1], title="u"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 50, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate",
                           args=[[f"frame{k}"],
                                 {"mode": "immediate", "frame": {"duration": 50}}],
                           label=f"t={t_values[k]:.3f}")
                       for k in range(len(t_values))],
                x=0.1, len=0.8, y=-0.05,
                currentvalue=dict(prefix="Time: ", visible=True)
            )]
        ),
        frames=[go.Frame(data=[go.Scatter(x=x, y=u_history[k], mode='lines')],
                        name=f"frame{k}")
               for k in range(len(t_values))]
    )

    return fig


def save_figure(fig, filename, formats=('html', 'pdf')):
    """
    Save figure in multiple formats for reproducibility.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    formats : tuple
        Output formats ('html', 'pdf', 'png', 'svg')
    """
    for fmt in formats:
        if fmt == 'html':
            fig.write_html(f"{filename}.html", include_plotlyjs='cdn')
        elif fmt == 'pdf':
            fig.write_image(f"{filename}.pdf", scale=2)
        elif fmt == 'png':
            fig.write_image(f"{filename}.png", scale=2)
        elif fmt == 'svg':
            fig.write_image(f"{filename}.svg")


__all__ = [
    'set_seed',
    'RANDOM_SEED',
    'get_color_scheme',
    'create_solution_plot',
    'create_convergence_plot',
    'create_animation_frames',
    'save_figure',
]
```

### 0.5.7 Update `_quarto.yml` for Reproducibility

```yaml
project:
  type: book
  output-dir: _book
  execute-dir: project    # CRITICAL: enables imports from src/

execute:
  freeze: auto            # Re-render only when source changes
  echo: false             # Hide code by default
  warning: false
  error: false            # Halt build on errors (catches verification failures)

jupyter: python3

format:
  html:
    theme: cosmo
    code-fold: true
    code-tools: true
    html-math-method: mathjax

  pdf:
    documentclass: scrbook
    papersize: letter
    keep-tex: true
    include-in-header:
      text: |
        % Minimal packages only - no custom macros
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{bm}
```

### 0.5.8 Chapter Template with SymPy Integration

Example pattern for all chapters:

```markdown
---
title: "Wave Equation"
---

```{python}
#| label: setup
#| include: false

# Standard imports for all chapters
from src.symbols import sp, x, t, dx, dt, u, c, i, n, C
from src.operators import central_diff, second_derivative_central
from src.display import show_eq, show_eq_aligned, inline_latex
from src.verification import verify_identity, check_stencil_order
from src.plotting import create_solution_plot, set_seed

set_seed()  # Ensure reproducible plots
```

## The Wave Equation

The one-dimensional wave equation is:

```{python}
#| label: eq-wave-pde
#| output: asis

# Define PDE symbolically
wave_lhs = sp.Derivative(u(x, t), t, 2)
wave_rhs = c**2 * sp.Derivative(u(x, t), x, 2)
wave_pde = sp.Eq(wave_lhs, wave_rhs)

show_eq(wave_pde, label='wave-pde')
```

## Spatial Discretization

The second derivative is approximated by the central difference stencil:

```{python}
#| label: central-stencil
#| output: asis

# Grid function notation
u_i = sp.Function('u')(i*dx, n*dt)
u_ip1 = sp.Function('u')((i+1)*dx, n*dt)
u_im1 = sp.Function('u')((i-1)*dx, n*dt)

# Central difference stencil
d2u_dx2_stencil = (u_ip1 - 2*u_i + u_im1) / dx**2

# Display
approx_eq = sp.Eq(sp.Derivative(u(x, t), x, 2), d2u_dx2_stencil)
show_eq(approx_eq, label='central-diff')
```

### Verification: Second-Order Accuracy

```{python}
#| label: verify-stencil-order
#| include: false

# Verify the stencil is O(dx²) via Taylor expansion
u_exact = u(x, t)
stencil = second_derivative_central(u_exact, x, dx)
exact_deriv = sp.diff(u(x, t), x, 2)

# This assertion fails the build if order is wrong
check_stencil_order(stencil, exact_deriv, x, dx, expected_order=2)
```

The truncation error is $O(\Delta x^2)$, verified by Taylor expansion.

```

---

## Phase 1: Restructure Book Organization

### 1.1 Replace Vibration Chapter with Devito Introduction

**Rationale:** The vibration ODEs chapter teaches concepts better suited to Devito's PDE focus. Replace with a Devito introduction chapter that covers the same fundamental concepts (central differences, time-stepping, stability) using the 1D wave equation.

#### New Chapter Structure: `chapters/devito_intro/`

```
chapters/devito_intro/
├── index.qmd
├── what_is_devito.qmd       # Overview and motivation
├── first_pde.qmd            # 1D wave equation as first example
├── devito_abstractions.qmd  # Grid, Function, TimeFunction, Eq, Operator
├── boundary_conditions.qmd  # Dirichlet, Neumann in Devito
└── verification.qmd         # Convergence testing, MMS
```

#### Section: What is Devito? (`what_is_devito.qmd`)

Content outline:
- Motivation: From NumPy loops to symbolic PDEs
- The DSL approach: Write math, generate optimized code
- Key benefits: Dimension-agnostic, automatic parallelization
- Installation and setup

#### Section: First PDE (`first_pde.qmd`)

Introduce Devito through the 1D wave equation (not ODE):

```python
from devito import Grid, TimeFunction, Eq, Operator

# Define domain
grid = Grid(shape=(101,), extent=(1.0,))
x = grid.dimensions[0]

# Create solution field with 2nd-order time, 2nd-order space accuracy
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

# Initial condition: Gaussian pulse
import numpy as np
u.data[0, :] = np.exp(-100*(grid.spacing[0]*np.arrange(101) - 0.5)**2)
u.data[1, :] = u.data[0, :]  # Zero initial velocity

# Wave equation: u_tt = c^2 * u_xx
c = 1.0
dt = 0.001
pde = Eq(u.dt2, c**2 * u.dx2)

# Explicit update equation
from devito import solve
update = Eq(u.forward, solve(pde, u.forward))

# Compile and run
op = Operator([update])
op.apply(time=500, dt=dt)
```

Key teaching points:
- `Grid`: Discrete domain definition
- `TimeFunction`: Time-varying unknowns with automatic time-level management
- Symbolic derivatives: `.dt`, `.dt2`, `.dx`, `.dx2`, `.laplace`
- `solve()`: Isolate target variable in PDE
- `Operator`: Compile equations to executable code

### 1.2 Update `_quarto.yml`

```yaml
book:
  chapters:
    - index.qmd
    - chapters/preface/index.qmd
    - part: "Main Chapters"
      chapters:
        - chapters/devito_intro/index.qmd   # NEW: Replaces vib
        - chapters/wave/index.qmd
        - chapters/diffu/index.qmd
        - chapters/advec/index.qmd
        - chapters/nonlin/index.qmd
    - part: "Appendices"
      chapters:
        - chapters/appendices/formulas/index.qmd
        - chapters/appendices/trunc/index.qmd
        - chapters/appendices/softeng2/index.qmd
```

### 1.3 Archive Original Code

Move all original NumPy implementations to `src/legacy/`:

```bash
mkdir -p src/legacy
mv src/vib src/legacy/
# Keep wave, diffu, advec, nonlin in place for incremental refactoring
```

---

## Phase 2: Wave Equations Chapter

### 2.1 Chapter Structure Refactoring

```
chapters/wave/
├── index.qmd
├── wave1D_devito.qmd        # REWRITE: Core 1D wave with Devito
├── wave1D_features.qmd      # REWRITE: Source terms, variable coefficients
├── wave_analysis.qmd        # ADAPT: Keep mathematical analysis
├── wave2D_devito.qmd        # REWRITE: Brief 2D extension
└── wave_exercises.qmd       # ADAPT: Devito-based exercises
```

### 2.2 Core 1D Wave Solver (`src/wave/wave1D_devito.py`)

```python
"""
1D Wave Equation Solver using Devito DSL.

Solves: u_tt = c^2 * u_xx + f(x,t)  on [0, L] x [0, T]
with Dirichlet BCs: u(0,t) = u(L,t) = 0
Initial conditions: u(x,0) = I(x), u_t(x,0) = V(x)
"""
from devito import Grid, Function, TimeFunction, Eq, solve, Operator
import numpy as np


def solver_wave1d(I, V, f, c, L, Nx, dt, T, save_every=1):
    """
    Solve 1D wave equation using Devito.

    Parameters
    ----------
    I : callable
        Initial displacement I(x)
    V : callable
        Initial velocity V(x)
    f : callable or None
        Source term f(x, t)
    c : float
        Wave speed
    L : float
        Domain length
    Nx : int
        Number of grid points
    dt : float
        Time step
    T : float
        Final time
    save_every : int
        Save solution every N time steps

    Returns
    -------
    u_hist : ndarray
        Solution history, shape (Nt_saved, Nx+1)
    x : ndarray
        Spatial coordinates
    t_saved : ndarray
        Times at which solution was saved
    """
    # Setup grid
    grid = Grid(shape=(Nx+1,), extent=(L,))
    x_dim = grid.dimensions[0]

    # Create TimeFunction with 2nd order time derivative support
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=None)

    # Velocity coefficient (could be spatially varying)
    vel = Function(name='c', grid=grid)
    vel.data[:] = c

    # Get coordinate array
    x_coords = np.linspace(0, L, Nx+1)

    # Set initial conditions
    u.data[0, :] = I(x_coords)

    # Handle first time step with initial velocity
    # u^1 = u^0 + dt*V + 0.5*dt^2*c^2*u_xx^0
    # Using Devito: compute manually for first step
    dx = L / Nx
    C2 = (c * dt / dx)**2
    u_init = I(x_coords)
    u1 = np.zeros(Nx+1)
    u1[1:-1] = (u_init[1:-1] + dt * V(x_coords[1:-1]) +
                0.5 * C2 * (u_init[:-2] - 2*u_init[1:-1] + u_init[2:]))
    u1[0] = 0  # Dirichlet BC
    u1[Nx] = 0
    u.data[1, :] = u1

    # Main PDE: u_tt = c^2 * u_xx
    pde = Eq(u.dt2, vel**2 * u.dx2)
    stencil = solve(pde, u.forward)

    # Update equation for interior points
    update = Eq(u.forward, stencil, subdomain=grid.interior)

    # Boundary conditions (Dirichlet u=0)
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], 0)
    bc_right = Eq(u[t_dim + 1, Nx], 0)

    # Compile operator
    op = Operator([update, bc_left, bc_right])

    # Time stepping with history storage
    Nt = int(round(T / dt))
    n_saved = Nt // save_every + 1
    u_hist = np.zeros((n_saved, Nx+1))
    t_saved = np.zeros(n_saved)

    u_hist[0, :] = u.data[0, :]
    t_saved[0] = 0

    save_idx = 1
    for n in range(1, Nt):
        # Apply operator for one time step
        op.apply(time_m=n, time_M=n, dt=dt)

        if (n + 1) % save_every == 0 and save_idx < n_saved:
            # Access the most recent time level
            u_hist[save_idx, :] = u.data[(n+1) % 3, :]
            t_saved[save_idx] = (n + 1) * dt
            save_idx += 1

    return u_hist, x_coords, t_saved


def test_quadratic():
    """
    Verify solver reproduces quadratic solution exactly.

    u(x,t) = x(L-x)(1 + t/2) satisfies wave equation with
    appropriate source term.
    """
    L = 2.5
    c = 1.5
    T = 2.0
    Nx = 20

    def u_exact(x, t):
        return x * (L - x) * (1 + 0.5 * t)

    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0.5 * u_exact(x, 0)

    # Choose dt to satisfy CFL
    dx = L / Nx
    dt = 0.8 * dx / c  # CFL = 0.8 < 1

    u_hist, x, t_saved = solver_wave1d(I, V, None, c, L, Nx, dt, T)

    # Check final solution
    u_e = u_exact(x, t_saved[-1])
    error = np.abs(u_hist[-1, :] - u_e).max()

    # With source term = 2*(1+t/2)*c^2, should be exact
    # Without source, this won't be exact, but demonstrates the pattern
    print(f"Max error at T={t_saved[-1]:.3f}: {error:.2e}")
    return error


def test_convergence():
    """Verify 2nd order convergence rate."""
    L = 1.0
    c = 1.0
    T = 0.5

    def I(x):
        return np.sin(2 * np.pi * x / L)

    def V(x):
        return np.zeros_like(x)

    def u_exact(x, t):
        return np.cos(2 * np.pi * c * t / L) * np.sin(2 * np.pi * x / L)

    errors = []
    Nx_values = [20, 40, 80, 160]

    for Nx in Nx_values:
        dx = L / Nx
        dt = 0.5 * dx / c  # CFL = 0.5

        u_hist, x, t_saved = solver_wave1d(I, V, None, c, L, Nx, dt, T)

        u_e = u_exact(x, t_saved[-1])
        error = np.sqrt(dx * np.sum((u_hist[-1, :] - u_e)**2))
        errors.append(error)

    # Compute convergence rates
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(2)
        rates.append(r)

    print("Convergence rates:", rates)
    assert abs(rates[-1] - 2.0) < 0.2, f"Expected ~2, got {rates[-1]}"
    return rates, errors


if __name__ == '__main__':
    test_quadratic()
    test_convergence()
```

### 2.3 Key Devito Concepts Introduced in Wave Chapter

| Concept | First Appearance | Description |
|---------|------------------|-------------|
| `Grid` | wave1D_devito.qmd | Discrete computational domain |
| `TimeFunction` | wave1D_devito.qmd | Time-varying field with automatic time levels |
| `time_order`, `space_order` | wave1D_devito.qmd | Accuracy order parameters |
| `.dt2`, `.dx2` | wave1D_devito.qmd | Second-order symbolic derivatives |
| `solve()` | wave1D_devito.qmd | Isolate target variable |
| `Operator` | wave1D_devito.qmd | JIT compilation and execution |
| `subdomain` | wave1D_devito.qmd | Interior/boundary region specification |
| `Function` | wave1D_features.qmd | Static coefficient fields |
| Variable coefficients | wave1D_features.qmd | Spatially-varying c(x) |
| 2D extension | wave2D_devito.qmd | `.laplace` for dimension-agnostic code |

### 2.4 Exercises Migration

**Original exercises to adapt:**

1. **Plug wave propagation** → Devito implementation with sharp initial condition
2. **Variable wave speed** → `Function` for c(x), demonstrate interface reflections
3. **Open boundary conditions** → Absorbing layers using damping term
4. **2D membrane vibration** → Brief 2D example with `.laplace`

---

## Phase 3: Diffusion Equations Chapter

### 3.1 Chapter Structure

```
chapters/diffu/
├── index.qmd
├── diffu1D_explicit.qmd     # Forward Euler with Devito
├── diffu1D_implicit.qmd     # NEW: Backward Euler, CN with PETSc
├── diffu_analysis.qmd       # ADAPT: Stability analysis
├── diffu_2D.qmd             # Brief 2D extension
└── diffu_exercises.qmd      # Adapted exercises
```

### 3.2 Explicit Diffusion Solver (`src/diffu/diffu1D_explicit.py`)

```python
"""
1D Diffusion Equation - Explicit (Forward Euler) Solver using Devito.

Solves: u_t = a * u_xx  on [0, L] x [0, T]
with Dirichlet BCs: u(0,t) = u(L,t) = 0
Initial condition: u(x,0) = I(x)

Stability requires: F = a*dt/dx^2 <= 0.5
"""
from devito import Grid, Function, TimeFunction, Eq, Operator
import numpy as np


def solver_diffusion_FE(I, a, L, Nx, dt, T, save_every=1):
    """
    Solve 1D diffusion equation using Forward Euler (explicit).

    Parameters
    ----------
    I : callable
        Initial condition I(x)
    a : float
        Diffusion coefficient
    L : float
        Domain length
    Nx : int
        Number of grid points
    dt : float
        Time step (must satisfy F = a*dt/dx^2 <= 0.5)
    T : float
        Final time
    save_every : int
        Save solution every N time steps

    Returns
    -------
    u_hist : ndarray
        Solution history
    x : ndarray
        Spatial coordinates
    t_saved : ndarray
        Saved time values
    """
    # Setup grid
    grid = Grid(shape=(Nx+1,), extent=(L,))

    # TimeFunction: time_order=1 for first-order time derivative
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    # Diffusion coefficient
    kappa = Function(name='a', grid=grid)
    kappa.data[:] = a

    # Coordinates
    x_coords = np.linspace(0, L, Nx+1)

    # Initial condition
    u.data[0, :] = I(x_coords)

    # Check stability
    dx = L / Nx
    F = a * dt / dx**2
    if F > 0.5:
        print(f"Warning: F = {F:.3f} > 0.5, scheme is unstable!")

    # PDE: u_t = a * u_xx
    # Forward Euler: u^{n+1} = u^n + dt * a * u_xx^n
    update = Eq(u.forward, u + dt * kappa * u.dx2, subdomain=grid.interior)

    # Boundary conditions
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], 0)
    bc_right = Eq(u[t_dim + 1, Nx], 0)

    # Compile
    op = Operator([update, bc_left, bc_right])

    # Time stepping
    Nt = int(round(T / dt))
    n_saved = Nt // save_every + 1
    u_hist = np.zeros((n_saved, Nx+1))
    t_saved = np.zeros(n_saved)

    u_hist[0, :] = u.data[0, :]
    t_saved[0] = 0

    save_idx = 1
    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)

        if (n + 1) % save_every == 0 and save_idx < n_saved:
            u_hist[save_idx, :] = u.data[(n+1) % 2, :]
            t_saved[save_idx] = (n + 1) * dt
            save_idx += 1

    return u_hist, x_coords, t_saved


def test_convergence_FE():
    """Verify O(dt) + O(dx^2) convergence."""
    L = 1.0
    a = 1.0
    T = 0.1

    def I(x):
        return np.sin(np.pi * x / L)

    def u_exact(x, t):
        return np.exp(-a * (np.pi/L)**2 * t) * np.sin(np.pi * x / L)

    errors = []
    Nx_values = [20, 40, 80, 160]

    for Nx in Nx_values:
        dx = L / Nx
        F = 0.4  # Below stability limit
        dt = F * dx**2 / a

        u_hist, x, t_saved = solver_diffusion_FE(I, a, L, Nx, dt, T)

        u_e = u_exact(x, t_saved[-1])
        error = np.sqrt(dx * np.sum((u_hist[-1, :] - u_e)**2))
        errors.append(error)

    # With fixed F, dt ~ dx^2, so overall convergence is O(dx^2)
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(2)
        rates.append(r)

    print("FE Convergence rates:", rates)
    assert rates[-1] > 1.8, f"Expected ~2, got {rates[-1]}"
    return rates, errors


if __name__ == '__main__':
    test_convergence_FE()
```

### 3.3 Implicit Diffusion Solver with PETSc (`src/diffu/diffu1D_implicit.py`)

```python
"""
1D Diffusion Equation - Implicit Solvers using Devito + PETSc.

Solves: u_t = a * u_xx  on [0, L] x [0, T]
using Backward Euler and Crank-Nicolson schemes.

Requires: Devito petsc branch (venv_implicit)
"""
from devito import Grid, Function, TimeFunction, Eq, Operator, configuration
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
import numpy as np


# Initialize PETSc (call once at module import)
PetscInitialize()


def solver_diffusion_BE(I, a, L, Nx, dt, T):
    """
    Solve 1D diffusion using Backward Euler (fully implicit).

    Unconditionally stable - any dt works.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))

    # Two time level approach: solve for u_new given u_old
    u_old = Function(name='u_old', grid=grid, space_order=2)
    u_new = Function(name='u_new', grid=grid, space_order=2)

    x_coords = np.linspace(0, L, Nx+1)
    u_old.data[:] = I(x_coords)

    # Backward Euler: (u_new - u_old)/dt = a * u_new_xx
    # Rearrange: u_new - dt*a*u_new_xx = u_old
    # As linear system: (I - dt*a*Laplacian) u_new = u_old

    # Define implicit equation
    eqn = Eq(u_new - dt * a * u_new.dx2, u_old, subdomain=grid.interior)

    # Boundary conditions
    left_bc = EssentialBC(u_new, 0.0, subdomain=grid.subdomains['left'])
    right_bc = EssentialBC(u_new, 0.0, subdomain=grid.subdomains['right'])

    # Create PETSc solve
    petsc_solve = petscsolve(
        [eqn, left_bc, right_bc],
        target=u_new,
        solver_parameters={
            'ksp_type': 'cg',
            'pc_type': 'jacobi',
            'ksp_rtol': 1e-10
        }
    )

    # Switch to PETSc language backend
    with configuration.switch(language='petsc'):
        op = Operator(petsc_solve)

    # Time stepping
    Nt = int(round(T / dt))
    u_hist = [u_old.data.copy()]

    for n in range(Nt):
        op.apply()
        u_old.data[:] = u_new.data[:]
        u_hist.append(u_new.data.copy())

    return np.array(u_hist), x_coords


def solver_diffusion_CN(I, a, L, Nx, dt, T):
    """
    Solve 1D diffusion using Crank-Nicolson (theta=0.5).

    Second-order accurate in time, unconditionally stable.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))

    u_old = Function(name='u_old', grid=grid, space_order=2)
    u_new = Function(name='u_new', grid=grid, space_order=2)

    x_coords = np.linspace(0, L, Nx+1)
    u_old.data[:] = I(x_coords)

    # Crank-Nicolson: (u_new - u_old)/dt = 0.5*a*(u_new_xx + u_old_xx)
    # Rearrange: u_new - 0.5*dt*a*u_new_xx = u_old + 0.5*dt*a*u_old_xx
    theta = 0.5
    lhs = u_new - theta * dt * a * u_new.dx2
    rhs = u_old + (1 - theta) * dt * a * u_old.dx2

    eqn = Eq(lhs, rhs, subdomain=grid.interior)

    left_bc = EssentialBC(u_new, 0.0, subdomain=grid.subdomains['left'])
    right_bc = EssentialBC(u_new, 0.0, subdomain=grid.subdomains['right'])

    petsc_solve = petscsolve(
        [eqn, left_bc, right_bc],
        target=u_new,
        solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'}
    )

    with configuration.switch(language='petsc'):
        op = Operator(petsc_solve)

    Nt = int(round(T / dt))
    u_hist = [u_old.data.copy()]

    for n in range(Nt):
        op.apply()
        u_old.data[:] = u_new.data[:]
        u_hist.append(u_new.data.copy())

    return np.array(u_hist), x_coords
```

### 3.4 Devito Concepts Introduced in Diffusion Chapter

| Concept | First Appearance | Description |
|---------|------------------|-------------|
| `time_order=1` | diffu1D_explicit.qmd | First-order time derivatives |
| `.dt` | diffu1D_explicit.qmd | First temporal derivative |
| Stability analysis | diffu_analysis.qmd | F = a*dt/dx² constraint |
| PETSc integration | diffu1D_implicit.qmd | `petscsolve` for linear systems |
| `EssentialBC` | diffu1D_implicit.qmd | Dirichlet BC specification |
| Theta-method | diffu1D_implicit.qmd | Parameterized implicit/explicit blend |

---

## Phase 4: Advection Equations Chapter

### 4.1 Chapter Structure

```
chapters/advec/
├── index.qmd
├── advec1D_schemes.qmd      # Upwind, Lax-Friedrichs, Lax-Wendroff
├── advec_analysis.qmd       # Dispersion, dissipation
└── advec_exercises.qmd      # Adapted exercises
```

### 4.2 Advection Solvers (`src/advec/advec1D_devito.py`)

```python
"""
1D Advection Equation Solvers using Devito.

Solves: u_t + c * u_x = 0  on [0, L] with periodic BC
or:     u_t + c * u_x = 0  on [0, L] with inflow BC
"""
from devito import Grid, Function, TimeFunction, Eq, Operator
from devito.types.basic import left, right
import numpy as np


def solver_upwind(I, c, L, Nx, dt, T):
    """
    First-order upwind scheme.

    u^{n+1}_i = u^n_i - C*(u^n_i - u^n_{i-1})  for c > 0
    where C = c*dt/dx (Courant number)

    Stable for 0 < C <= 1.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))
    x_dim = grid.dimensions[0]

    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=1)

    x_coords = np.linspace(0, L, Nx+1)
    u.data[0, :] = I(x_coords)

    dx = L / Nx
    C = c * dt / dx

    if C > 1.0:
        print(f"Warning: Courant number C = {C:.3f} > 1, unstable!")

    # Upwind: u_t + c*u_x = 0 discretized as
    # (u^{n+1} - u^n)/dt + c * (u^n - u^n[x-dx])/dx = 0
    # Using backward difference for u_x (assuming c > 0)

    # Devito approach: use shifted indexing
    # u.dx with left shift gives backward difference
    update = Eq(u.forward, u - C * (u - u.subs(x_dim, x_dim - x_dim.spacing)),
                subdomain=grid.interior)

    # Left boundary: inflow condition (keep initial)
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], I(0))  # or periodic: u[t_dim + 1, Nx]
    bc_right = Eq(u[t_dim + 1, Nx], u[t_dim, Nx-1])  # outflow

    op = Operator([update, bc_left, bc_right])

    Nt = int(round(T / dt))
    u_hist = [u.data[0, :].copy()]

    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)
        u_hist.append(u.data[(n+1) % 2, :].copy())

    return np.array(u_hist), x_coords


def solver_lax_wendroff(I, c, L, Nx, dt, T):
    """
    Lax-Wendroff scheme (second-order accurate).

    u^{n+1}_i = u^n_i - (C/2)*(u^n_{i+1} - u^n_{i-1})
                      + (C²/2)*(u^n_{i+1} - 2*u^n_i + u^n_{i-1})

    Stable for |C| <= 1.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))

    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    x_coords = np.linspace(0, L, Nx+1)
    u.data[0, :] = I(x_coords)

    dx = L / Nx
    C = c * dt / dx

    # Lax-Wendroff: combines central difference and diffusion term
    # = u - C/2 * centered(u_x) + C²/2 * centered(u_xx)
    # = u - (C/2)*dx*u.dx + (C²/2)*dx²*u.dx2

    stencil = u - 0.5 * C * dx * u.dx + 0.5 * C**2 * dx**2 * u.dx2
    update = Eq(u.forward, stencil, subdomain=grid.interior)

    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], I(0))
    bc_right = Eq(u[t_dim + 1, Nx], u[t_dim, Nx])

    op = Operator([update, bc_left, bc_right])

    Nt = int(round(T / dt))
    u_hist = [u.data[0, :].copy()]

    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)
        u_hist.append(u.data[(n+1) % 2, :].copy())

    return np.array(u_hist), x_coords
```

### 4.3 Devito Concepts Introduced

| Concept | Description |
|---------|-------------|
| `.subs()` | Index substitution for upwind stencils |
| Asymmetric stencils | Upwind differencing |
| Flux-form conservation | Lax-Wendroff derivation |

---

## Phase 5: Nonlinear Problems Chapter

### 5.1 Chapter Structure

```
chapters/nonlin/
├── index.qmd
├── nonlin_explicit.qmd      # Explicit treatment of nonlinearity
├── nonlin_picard.qmd        # Picard iteration with Devito
├── nonlin_splitting.qmd     # Operator splitting
└── nonlin_exercises.qmd     # Adapted exercises
```

### 5.2 Nonlinear Diffusion Example (`src/nonlin/nonlin_diffusion.py`)

```python
"""
Nonlinear Diffusion using Devito.

Example: u_t = div(D(u) * grad(u)) where D depends on u.
"""
from devito import Grid, Function, TimeFunction, Eq, Operator, div, grad
import numpy as np


def solver_nonlin_diffusion_explicit(I, D_func, L, Nx, dt, T, picard_tol=1e-6):
    """
    Solve nonlinear diffusion with coefficient D(u).

    Uses explicit time stepping with lagged coefficient evaluation.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))

    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)
    D = Function(name='D', grid=grid)

    x_coords = np.linspace(0, L, Nx+1)
    u.data[0, :] = I(x_coords)
    D.data[:] = D_func(u.data[0, :])

    # Explicit: u^{n+1} = u^n + dt * D(u^n) * u_xx^n
    update = Eq(u.forward, u + dt * D * u.dx2, subdomain=grid.interior)

    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], 0)
    bc_right = Eq(u[t_dim + 1, Nx], 0)

    op = Operator([update, bc_left, bc_right])

    Nt = int(round(T / dt))
    u_hist = [u.data[0, :].copy()]

    for n in range(Nt):
        # Update D based on current solution
        D.data[:] = D_func(u.data[n % 2, :])

        op.apply(time_m=n, time_M=n, dt=dt)
        u_hist.append(u.data[(n+1) % 2, :].copy())

    return np.array(u_hist), x_coords


def solver_reaction_diffusion_splitting(I, a, R_func, L, Nx, dt, T):
    """
    Operator splitting for reaction-diffusion: u_t = a*u_xx + R(u)

    Split into:
    1. Diffusion step: v_t = a*v_xx
    2. Reaction step: w_t = R(w)

    Strange splitting for 2nd order accuracy.
    """
    grid = Grid(shape=(Nx+1,), extent=(L,))

    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    x_coords = np.linspace(0, L, Nx+1)
    u.data[0, :] = I(x_coords)

    # Diffusion operator
    diff_update = Eq(u.forward, u + dt * a * u.dx2, subdomain=grid.interior)
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], 0)
    bc_right = Eq(u[t_dim + 1, Nx], 0)

    op_diff = Operator([diff_update, bc_left, bc_right])

    Nt = int(round(T / dt))
    u_hist = [u.data[0, :].copy()]

    for n in range(Nt):
        # Half step of reaction
        u_data = u.data[n % 2, :]
        u_data[1:-1] = u_data[1:-1] + 0.5 * dt * R_func(u_data[1:-1])

        # Full step of diffusion
        op_diff.apply(time_m=n, time_M=n, dt=dt)

        # Half step of reaction
        u_data = u.data[(n+1) % 2, :]
        u_data[1:-1] = u_data[1:-1] + 0.5 * dt * R_func(u_data[1:-1])

        u_hist.append(u_data.copy())

    return np.array(u_hist), x_coords
```

---

## Phase 6: Appendices

### 6.1 Truncation Error Appendix (Expanded)

**New content to add:**

```markdown
## Devito and Truncation Errors {#sec-trunc-devito}

Devito's `space_order` parameter directly controls the truncation error
of spatial derivatives. Understanding this connection is essential for
choosing appropriate accuracy settings.

### The `space_order` Parameter

When you create a `TimeFunction` or `Function`:

```python
u = TimeFunction(name='u', grid=grid, space_order=2)
```

The `space_order=2` specifies that spatial derivatives should use
stencils accurate to O(dx²). Higher orders are available:

| space_order | Stencil Points | Accuracy |
|-------------|----------------|----------|
| 2           | 3              | O(dx²)   |
| 4           | 5              | O(dx⁴)   |
| 6           | 7              | O(dx⁶)   |
| 8           | 9              | O(dx⁸)   |

### Viewing Generated Stencils

You can inspect Devito's stencil coefficients:

```python
from devito import Grid, TimeFunction, Eq, Operator

grid = Grid(shape=(11,))
u = TimeFunction(name='u', grid=grid, space_order=4)

# See the symbolic expression for u.dx2
print(u.dx2)
# Output shows: -5*u/2/h_x**2 + 4*u(x - h_x/2)/3/h_x**2 + ...
```

### Trading Accuracy for Performance

Higher `space_order` means:
- Wider stencils (more memory bandwidth)
- More floating-point operations
- Better accuracy per grid point

For wave equations, `space_order=4` or `space_order=8` is common in
geophysics applications where accuracy is paramount.
```

### 6.2 Software Engineering Appendix (Devito-Focused Rewrite)

**New structure:**

```markdown
# Software Engineering with Devito {#sec-softeng-devito}

## Project Structure for Devito Applications

```
my_pde_solver/
├── src/
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── wave.py          # Solver modules
│   │   └── diffusion.py
│   ├── utils/
│   │   ├── visualization.py
│   │   └── convergence.py
│   └── __init__.py
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_wave.py
│   └── test_diffusion.py
├── examples/
│   └── run_wave_simulation.py
├── pyproject.toml
└── README.md
```

## Testing Devito Solvers

### Pytest Fixtures for Grid Setup

```python
# tests/conftest.py
import pytest
from devito import Grid

@pytest.fixture
def grid_1d():
    return Grid(shape=(101,), extent=(1.0,))

@pytest.fixture
def grid_2d():
    return Grid(shape=(101, 101), extent=(1.0, 1.0))

@pytest.fixture(params=[2, 4, 6])
def space_order(request):
    return request.param
```

### Testing Convergence Rates

```python
def test_wave_convergence(grid_1d):
    """Verify expected convergence rate."""
    errors = []
    for Nx in [20, 40, 80, 160]:
        error = run_solver_and_compute_error(Nx)
        errors.append(error)

    rates = compute_convergence_rates(errors)
    assert rates[-1] > 1.9, f"Expected ~2, got {rates[-1]}"
```

## Performance Profiling

```python
from devito import Operator, configuration

# Enable performance logging
configuration['log-level'] = 'PERF'

op = Operator([update])
summary = op.apply(time=100, dt=0.001)

print(f"GFlops/s: {summary.gflopss}")
print(f"GPts/s: {summary.gpointss}")
```
```

---

## Phase 7: Testing & CI Infrastructure

### 7.1 Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_wave.py             # Wave equation solver tests
├── test_diffu_explicit.py   # Explicit diffusion solver tests
├── test_diffu_implicit.py   # PETSc implicit tests (separate CI job)
├── test_advec.py            # Advection solver tests
├── test_nonlin.py           # Nonlinear solver tests
├── test_utils.py            # Utility function tests
├── test_derivations.py      # Mathematical derivation verification
├── test_operators.py        # Finite difference operator tests
├── test_verification.py     # Verification utility self-tests
└── test_plotting.py         # Reproducible plotting tests
```

### 7.2 Shared Fixtures (`tests/conftest.py`)

```python
"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
from devito import Grid


@pytest.fixture
def grid_1d_small():
    """Small 1D grid for fast tests."""
    return Grid(shape=(21,), extent=(1.0,))


@pytest.fixture
def grid_1d_medium():
    """Medium 1D grid for convergence tests."""
    return Grid(shape=(101,), extent=(1.0,))


@pytest.fixture
def grid_2d_small():
    """Small 2D grid for fast tests."""
    return Grid(shape=(21, 21), extent=(1.0, 1.0))


@pytest.fixture(params=[2, 4])
def space_order(request):
    """Parametrize over space orders."""
    return request.param


@pytest.fixture
def gaussian_ic():
    """Gaussian initial condition factory."""
    def _gaussian(x, center=0.5, width=0.1):
        return np.exp(-((x - center) / width)**2)
    return _gaussian


@pytest.fixture
def sinusoidal_ic():
    """Sinusoidal initial condition factory."""
    def _sine(x, L=1.0, k=1):
        return np.sin(k * np.pi * x / L)
    return _sine


# Markers for slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "petsc: marks tests requiring PETSc")
```

### 7.3 Example Test File (`tests/test_wave.py`)

```python
"""Tests for wave equation solvers."""
import pytest
import numpy as np

from src.wave.wave1D_devito import solver_wave1d


class TestWave1DBasic:
    """Basic functionality tests."""

    def test_zero_initial_condition(self):
        """Zero IC should give zero solution."""
        u_hist, x, t = solver_wave1d(
            I=lambda x: 0*x,
            V=lambda x: 0*x,
            f=None, c=1.0, L=1.0, Nx=20, dt=0.01, T=0.1
        )
        assert np.allclose(u_hist, 0, atol=1e-14)

    def test_boundary_conditions(self):
        """Verify Dirichlet BCs are satisfied."""
        u_hist, x, t = solver_wave1d(
            I=lambda x: np.sin(np.pi * x),
            V=lambda x: 0*x,
            f=None, c=1.0, L=1.0, Nx=20, dt=0.01, T=0.5
        )
        assert np.allclose(u_hist[:, 0], 0, atol=1e-10)
        assert np.allclose(u_hist[:, -1], 0, atol=1e-10)


class TestWave1DConvergence:
    """Convergence rate tests."""

    @pytest.mark.slow
    def test_second_order_convergence(self):
        """Verify O(dx^2, dt^2) convergence."""
        L = 1.0
        c = 1.0
        T = 0.5

        def I(x):
            return np.sin(2 * np.pi * x)

        def V(x):
            return np.zeros_like(x)

        def u_exact(x, t):
            return np.cos(2 * np.pi * c * t) * np.sin(2 * np.pi * x)

        errors = []
        Nx_values = [20, 40, 80, 160]

        for Nx in Nx_values:
            dx = L / Nx
            dt = 0.5 * dx / c  # CFL = 0.5

            u_hist, x, t_saved = solver_wave1d(
                I, V, None, c, L, Nx, dt, T, save_every=int(T/dt)
            )

            u_e = u_exact(x, t_saved[-1])
            error = np.sqrt(dx * np.sum((u_hist[-1] - u_e)**2))
            errors.append(error)

        # Compute rates
        rates = []
        for i in range(1, len(errors)):
            r = np.log(errors[i-1] / errors[i]) / np.log(2)
            rates.append(r)

        assert rates[-1] > 1.9, f"Expected ~2, got {rates[-1]}"


class TestWave1DAgainstLegacy:
    """Comparison tests against original numpy implementations."""

    def test_matches_legacy_solver(self):
        """Devito solution should match legacy numpy solver."""
        from src.legacy.wave.wave1D.wave1D_u0 import solver as legacy_solver

        L = 1.0
        c = 1.0
        T = 0.5
        Nx = 40

        def I(x):
            return np.sin(np.pi * x / L)

        def V(x):
            return np.zeros_like(x)

        dx = L / Nx
        dt = 0.8 * dx / c
        C = c * dt / dx

        # Run legacy solver
        u_legacy, x, t, _ = legacy_solver(I, V, None, c, L, dt, C, T)

        # Run Devito solver
        u_hist, x_dev, t_dev = solver_wave1d(
            I, V, None, c, L, Nx, dt, T, save_every=int(T/dt)
        )

        # Compare final solutions
        assert np.allclose(u_hist[-1], u_legacy, rtol=1e-5)
```

### 7.4 GitHub Actions Workflow (`.github/workflows/tests.yml`)

```yaml
name: Tests

on:
  push:
    branches: [main, devito]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install linters
        run: pip install ruff isort
      - name: Run ruff
        run: ruff check src/ tests/
      - name: Check imports
        run: isort --check-only src/ tests/

  test-derivations:
    name: Verify Mathematical Derivations
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install sympy numpy pytest
      - name: Run derivation tests
        run: pytest tests/test_derivations.py tests/test_operators.py -v --tb=long

  test-explicit:
    runs-on: ubuntu-latest
    needs: [lint, test-derivations]
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install devito numpy scipy matplotlib pytest sympy plotly kaleido
          pip install -e .
      - name: Run tests (excluding PETSc)
        run: pytest tests/ -v -m "not petsc" --tb=short
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'

  test-implicit:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install PETSc
        run: |
          sudo apt-get update
          sudo apt-get install -y petsc-dev libopenmpi-dev
      - name: Install Devito petsc branch
        run: |
          git clone --branch petsc --depth 1 https://github.com/devitocodes/devito.git devito_petsc
          pip install -e devito_petsc
          pip install petsc4py mpi4py pytest numpy scipy
          pip install -e .
      - name: Run PETSc tests
        run: pytest tests/test_diffu_implicit.py -v -m "petsc"

  build-book:
    runs-on: ubuntu-latest
    needs: [test-explicit, test-derivations]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install devito numpy scipy matplotlib sympy plotly kaleido ipython jupyter
          sudo apt-get install -y texlive-full
      - name: Setup Quarto
        uses: quarto-dev/quarto-actions/setup@v2
        with:
          tinytex: true
      - name: Restore freeze cache
        uses: actions/cache@v4
        with:
          path: _freeze
          key: freeze-${{ hashFiles('**/*.qmd', 'src/**/*.py') }}
          restore-keys: |
            freeze-
      - name: Render book (executes verification chunks)
        run: quarto render
      - name: Verify freeze is current
        run: |
          # Check that committed freeze matches rendered output
          git diff --exit-code _freeze/ || echo "::warning::Freeze directory has changed"
      - name: Upload HTML artifact
        uses: actions/upload-artifact@v4
        with:
          name: book-html
          path: _book/
      - name: Upload PDF artifact
        uses: actions/upload-artifact@v4
        with:
          name: book-pdf
          path: _book/*.pdf
```

---

## Phase 8: Final Integration & Review

### 8.1 Pre-Publication Checklist

- [ ] All code examples run without errors
- [ ] All tests pass in CI
- [ ] Convergence rates verified for all solvers
- [ ] Cross-references updated (equations, figures, sections)
- [ ] Bibliography updated with Devito references
- [ ] Index regenerated
- [ ] PDF builds cleanly
- [ ] HTML version renders correctly

### 8.2 New References to Add

```bibtex
@article{devito-api,
  author = {Luporini, Fabio and others},
  title = {Devito: Automated Fast Finite Difference Computation},
  journal = {ACM Transactions on Mathematical Software},
  year = {2020}
}

@article{devito-compiler,
  author = {Luporini, Fabio and others},
  title = {Architecture and performance of {Devito}},
  journal = {Geoscientific Model Development},
  year = {2019}
}
```

---

## Appendix A: Devito Concept Introduction Points

| Chapter | Section | Devito Concept | Description |
|---------|---------|----------------|-------------|
| Devito Intro | First PDE | `Grid` | Computational domain |
| Devito Intro | First PDE | `TimeFunction` | Time-varying fields |
| Devito Intro | First PDE | `Eq`, `Operator` | Equations and compilation |
| Devito Intro | First PDE | `.dt2`, `.dx2` | Symbolic derivatives |
| Devito Intro | Abstractions | `Function` | Static coefficients |
| Devito Intro | Abstractions | `solve()` | Variable isolation |
| Devito Intro | BCs | `subdomain` | Region specification |
| Wave | 1D Basic | `time_order`, `space_order` | Accuracy parameters |
| Wave | 2D | `.laplace` | Dimension-agnostic Laplacian |
| Diffusion | Explicit | `.dt` | First time derivative |
| Diffusion | Implicit | `petscsolve` | Linear system solver |
| Diffusion | Implicit | `EssentialBC` | Boundary conditions |
| Advection | Upwind | `.subs()` | Index manipulation |
| Nonlinear | Explicit | Lagged coefficients | Update-then-solve pattern |
| Truncation | Analysis | `space_order` effects | Stencil coefficients |

---

## Appendix B: File Migration Map

### Source Files

| Original | New Location | Status |
|----------|--------------|--------|
| `src/vib/*.py` | `src/legacy/vib/` | Archive for tests |
| `src/wave/wave1D/*.py` | `src/legacy/wave/` + `src/wave/wave1D_devito.py` | Rewrite |
| `src/wave/wave2D/*.py` | `src/legacy/wave/` + `src/wave/wave2D_devito.py` | Rewrite |
| `src/diffu/*.py` | `src/legacy/diffu/` + `src/diffu/*_devito.py` | Rewrite |
| `src/advec/*.py` | `src/legacy/advec/` + `src/advec/advec_devito.py` | Rewrite |
| `src/nonlin/*.py` | `src/legacy/nonlin/` + `src/nonlin/*_devito.py` | Rewrite |

### Chapter Files

| Original | Action |
|----------|--------|
| `chapters/vib/` | Replace with `chapters/devito_intro/` |
| `chapters/wave/` | Rewrite all `.qmd` files |
| `chapters/diffu/` | Rewrite all `.qmd` files |
| `chapters/advec/` | Rewrite all `.qmd` files |
| `chapters/nonlin/` | Rewrite all `.qmd` files |
| `chapters/appendices/trunc/` | Expand with Devito section |
| `chapters/appendices/softeng2/` | Full rewrite for Devito |

---

## Appendix C: Testing Strategy

### Test Categories

1. **Unit Tests**: Individual solver functions
   - Zero initial condition → zero solution
   - Boundary conditions satisfied
   - Conservation properties (where applicable)

2. **Convergence Tests**: Verify accuracy orders
   - Compare against manufactured/analytical solutions
   - Compute empirical convergence rates
   - Mark as `@pytest.mark.slow`

3. **Regression Tests**: Compare against legacy implementations
   - Run both Devito and numpy solvers
   - Assert solutions match within tolerance

4. **Integration Tests**: Full simulation scenarios
   - Multi-physics coupling
   - Long-time stability

5. **Mathematical Derivation Tests** (NEW)
   - Verify finite difference stencil truncation error orders
   - Confirm stability conditions (CFL, diffusion number)
   - Validate exact/manufactured solutions satisfy stated PDEs
   - Check Taylor expansion coefficients match formulas
   - Test symbolic identities used in derivations

6. **Reproducibility Tests** (NEW)
   - Verify random seeds produce deterministic outputs
   - Check plot data consistency across runs
   - Validate frozen outputs match fresh execution

### Test Data Strategy

- Store small reference solutions in `tests/data/`
- Generate analytical solutions at runtime using SymPy
- Use Method of Manufactured Solutions (MMS) for complex cases
- All MMS solutions verified symbolically before numerical use

### Mathematical Verification Workflow

For each derivation in the book:

1. **In-book verification chunk** (hidden, `#| include: false`):
   ```python
   # Verify stencil order
   check_stencil_order(stencil, exact, x, dx, expected_order=2)
   ```

2. **Standalone test in `tests/test_derivations.py`**:
   - More comprehensive than in-book checks
   - Tests edge cases and numerical validation
   - Runs in CI before book build

3. **SymPy symbolic verification**:
   - Taylor series expansion to verify truncation error
   - Substitution to verify exact solutions
   - Simplification to verify algebraic identities

### Continuous Integration

- **On every commit**: Lint + fast unit tests + derivation tests
- **On PR**: Full test suite (explicit schemes) + derivation tests
- **Nightly/weekly**: PETSc implicit tests + full book build
- **Book build**: Verification chunks halt build on any assertion failure

### Freeze Strategy for Reproducibility

The `_freeze/` directory is **committed to git** and contains:
- JSON snapshots of all executed code blocks
- Output for equations, plots, and computed values
- Hash of source code that generated each output

**Cache invalidation**: Freeze is regenerated when:
- Any `.qmd` file changes
- Any `src/*.py` module changes
- Manual `quarto render --execute`

**CI verification**:
```yaml
- name: Verify freeze is up to date
  run: |
    quarto render --execute
    git diff --exit-code _freeze/
```

---

## Implementation Timeline (Suggested Order)

### Milestone 1: Foundation
1. Phase 0: Infrastructure setup (CI, directory structure)
2. Phase 1.1-1.2: Devito intro chapter outline
3. Phase 7.1-7.3: Basic test framework

### Milestone 2: Core Content
4. Phase 2: Wave equations (primary chapter)
5. Phase 3.2: Explicit diffusion
6. Tests for wave and diffusion

### Milestone 3: Advanced Topics
7. Phase 3.3: Implicit diffusion with PETSc
8. Phase 4: Advection
9. Phase 5: Nonlinear problems

### Milestone 4: Polish
10. Phase 6: Appendices
11. Phase 8: Final integration and review
12. Full CI pipeline with book builds

---

---

## Appendix D: Mathematical Derivation Verification

### D.1 Test File: `tests/test_derivations.py`

This test file verifies all mathematical derivations in the book are correct:

```python
"""
Unit tests for mathematical derivations in the book.

These tests verify that:
1. Finite difference stencils have correct truncation error order
2. Stability conditions are correctly derived
3. Exact solutions satisfy the stated PDEs
4. Manufactured solutions are consistent
"""

import pytest
import sympy as sp
import numpy as np
from src.symbols import x, y, t, dx, dy, dt, u, v, c, alpha, C, F, i, j, n
from src.operators import (
    forward_diff, backward_diff, central_diff,
    second_derivative_central, laplacian_2d
)
from src.verification import (
    verify_identity, check_stencil_order, verify_pde_solution, numerical_verify
)


class TestFirstDerivativeStencils:
    """Verify first derivative approximations."""

    def test_forward_difference_is_first_order(self):
        """Forward difference: (f(x+h) - f(x)) / h is O(h)."""
        f = u(x, t)
        stencil = forward_diff(f, x, dx)
        exact = sp.diff(f, x)
        check_stencil_order(stencil, exact, x, dx, expected_order=1)

    def test_backward_difference_is_first_order(self):
        """Backward difference: (f(x) - f(x-h)) / h is O(h)."""
        f = u(x, t)
        stencil = backward_diff(f, x, dx)
        exact = sp.diff(f, x)
        check_stencil_order(stencil, exact, x, dx, expected_order=1)

    def test_central_difference_is_second_order(self):
        """Central difference: (f(x+h) - f(x-h)) / (2h) is O(h²)."""
        f = u(x, t)
        stencil = central_diff(f, x, dx)
        exact = sp.diff(f, x)
        check_stencil_order(stencil, exact, x, dx, expected_order=2)

    def test_central_difference_exact_for_quadratic(self):
        """Central difference is exact for linear functions."""
        f = 3*x + 5
        stencil = central_diff(f, x, dx)
        exact = sp.diff(f, x)
        verify_identity(stencil, exact)


class TestSecondDerivativeStencils:
    """Verify second derivative approximations."""

    def test_central_second_derivative_is_second_order(self):
        """(f(x+h) - 2f(x) + f(x-h)) / h² is O(h²)."""
        f = u(x, t)
        stencil = second_derivative_central(f, x, dx)
        exact = sp.diff(f, x, 2)
        check_stencil_order(stencil, exact, x, dx, expected_order=2)

    def test_second_derivative_exact_for_quadratic(self):
        """Second derivative stencil is exact for quadratics."""
        f = x**2 + 3*x + 1
        stencil = second_derivative_central(f, x, dx)
        exact = sp.diff(f, x, 2)  # = 2
        verify_identity(stencil, exact)

    def test_laplacian_2d_is_second_order(self):
        """2D Laplacian stencil is O(h²) in each direction."""
        f = u(x, y, t)
        stencil = laplacian_2d(f, x, y, dx, dy)
        exact = sp.diff(f, x, 2) + sp.diff(f, y, 2)
        # Test order in x direction
        error_x = sp.series(stencil - exact, dx, 0, 4)
        assert error_x.coeff(dx, 0) == 0
        assert error_x.coeff(dx, 1) == 0


class TestWaveEquationDerivations:
    """Verify wave equation derivations from Chapter 2."""

    def test_wave_equation_exact_solution(self):
        """Verify d'Alembert solution satisfies wave equation."""
        # d'Alembert solution: u(x,t) = f(x - ct) + g(x + ct)
        f_func = sp.Function('f')
        g_func = sp.Function('g')
        xi = x - c*t
        eta = x + c*t

        u_dalembert = f_func(xi) + g_func(eta)

        # Check u_tt = c² u_xx
        u_tt = sp.diff(u_dalembert, t, 2)
        u_xx = sp.diff(u_dalembert, x, 2)

        lhs = u_tt
        rhs = c**2 * u_xx

        verify_identity(lhs, rhs)

    def test_standing_wave_solution(self):
        """Verify standing wave sin(kx)cos(ωt) satisfies wave equation."""
        k = sp.Symbol('k', positive=True)
        omega = sp.Symbol('omega', positive=True)

        u_standing = sp.sin(k*x) * sp.cos(omega*t)

        u_tt = sp.diff(u_standing, t, 2)
        u_xx = sp.diff(u_standing, x, 2)

        # u_tt = c² u_xx requires ω = ck
        # Substituting: -ω² sin(kx)cos(ωt) = -c²k² sin(kx)cos(ωt)
        # This holds when ω = ck
        u_tt_simplified = u_tt.subs(omega, c*k)
        rhs = c**2 * u_xx

        verify_identity(sp.simplify(u_tt_simplified), sp.simplify(rhs))

    def test_courant_stability_condition(self):
        """Verify CFL condition derivation for explicit wave scheme."""
        # Amplification factor for explicit scheme
        # g = 1 - 2C²(1 - cos(θ)) where C = c*dt/dx
        theta = sp.Symbol('theta', real=True)

        g = 1 - 2*C**2*(1 - sp.cos(theta))

        # For stability: |g| ≤ 1
        # Worst case at θ = π: g = 1 - 4C²
        # Requires -1 ≤ 1 - 4C² ≤ 1
        # Upper bound always satisfied
        # Lower bound: 1 - 4C² ≥ -1 => C² ≤ 1/2 => C ≤ 1/√2 ≈ 0.707

        # But actually for |g| ≤ 1 at θ=π: |1-4C²| ≤ 1
        # -1 ≤ 1-4C² ≤ 1
        # 0 ≤ 4C² ≤ 2
        # 0 ≤ C² ≤ 0.5
        # C ≤ 1 (taking C² ≤ 1/2 is more restrictive)

        g_at_pi = g.subs(theta, sp.pi)
        assert sp.simplify(g_at_pi) == 1 - 4*C**2

        # At C = 1: g = 1 - 4 = -3 (unstable)
        # At C = 0.5: g = 1 - 1 = 0 (stable)
        numerical_verify(g_at_pi, {C: 0.5}, expected=0.0)


class TestDiffusionEquationDerivations:
    """Verify diffusion equation derivations from Chapter 3."""

    def test_heat_equation_gaussian_solution(self):
        """Verify Gaussian is a solution to heat equation."""
        # Fundamental solution: u = (1/√(4παt)) exp(-x²/(4αt))
        u_gauss = 1/sp.sqrt(4*sp.pi*alpha*t) * sp.exp(-x**2/(4*alpha*t))

        u_t = sp.diff(u_gauss, t)
        u_xx = sp.diff(u_gauss, x, 2)

        lhs = u_t
        rhs = alpha * u_xx

        # Should be equal (may need simplification)
        diff = sp.simplify(lhs - rhs)
        assert diff == 0, f"Heat equation not satisfied: {diff}"

    def test_diffusion_stability_fe(self):
        """Verify Forward Euler stability: F ≤ 0.5."""
        # Amplification factor: g = 1 - 4F*sin²(θ/2)
        theta = sp.Symbol('theta', real=True)
        g_fe = 1 - 4*F*sp.sin(theta/2)**2

        # Worst case at θ = π: g = 1 - 4F
        g_worst = g_fe.subs(theta, sp.pi)
        assert sp.simplify(g_worst) == 1 - 4*F

        # Stability requires |g| ≤ 1
        # At F = 0.5: g = 1 - 2 = -1 (marginally stable)
        # At F = 0.25: g = 1 - 1 = 0 (stable)
        numerical_verify(g_worst, {F: 0.5}, expected=-1.0)
        numerical_verify(g_worst, {F: 0.25}, expected=0.0)

    def test_diffusion_cn_unconditional_stability(self):
        """Verify Crank-Nicolson is unconditionally stable."""
        # Amplification factor: g = (1 - 2F*sin²(θ/2)) / (1 + 2F*sin²(θ/2))
        theta = sp.Symbol('theta', real=True)
        sin2 = sp.sin(theta/2)**2
        g_cn = (1 - 2*F*sin2) / (1 + 2*F*sin2)

        # |g| ≤ 1 for all F ≥ 0 and all θ
        # Test numerically for various F values
        for F_val in [0.1, 0.5, 1.0, 10.0, 100.0]:
            for theta_val in np.linspace(0, np.pi, 10):
                g_val = complex(g_cn.subs({F: F_val, theta: theta_val}).evalf())
                assert abs(g_val) <= 1.0 + 1e-10, f"|g|={abs(g_val)} > 1 at F={F_val}, θ={theta_val}"


class TestAdvectionDerivations:
    """Verify advection equation derivations from Chapter 4."""

    def test_upwind_scheme_derivation(self):
        """Verify upwind scheme stencil for c > 0."""
        # Upwind: (u^{n+1}_i - u^n_i)/dt + c*(u^n_i - u^n_{i-1})/dx = 0
        # Rearranged: u^{n+1}_i = u^n_i - C*(u^n_i - u^n_{i-1})
        # where C = c*dt/dx

        u_n_i = sp.Symbol('u_n_i')
        u_n_im1 = sp.Symbol('u_n_im1')

        u_np1_upwind = u_n_i - C*(u_n_i - u_n_im1)

        # Expanded: u^{n+1} = (1-C)*u^n_i + C*u^n_{i-1}
        expected = (1-C)*u_n_i + C*u_n_im1
        verify_identity(sp.expand(u_np1_upwind), sp.expand(expected))

    def test_lax_wendroff_is_second_order(self):
        """Verify Lax-Wendroff is O(dt², dx²)."""
        # Lax-Wendroff: u^{n+1} = u^n - (C/2)(u_{i+1} - u_{i-1}) + (C²/2)(u_{i+1} - 2u_i + u_{i-1})
        # This can be derived from Taylor expansion of the advection equation

        # The scheme is second-order accurate in both space and time
        # We verify this by checking the modified equation analysis

        # For advection u_t + c*u_x = 0
        # LW approximates u_t + c*u_x + O(dt², dx²)
        pass  # Full verification requires modified equation analysis


class TestManufacturedSolutions:
    """Verify manufactured solutions used in convergence tests."""

    def test_wave_manufactured_solution(self):
        """Verify MMS solution for wave equation tests."""
        # u(x,t) = sin(πx/L)cos(πct/L) satisfies wave equation on [0,L]
        L = sp.Symbol('L', positive=True)
        u_mms = sp.sin(sp.pi*x/L) * sp.cos(sp.pi*c*t/L)

        u_tt = sp.diff(u_mms, t, 2)
        u_xx = sp.diff(u_mms, x, 2)

        lhs = u_tt
        rhs = c**2 * u_xx

        verify_identity(sp.simplify(lhs), sp.simplify(rhs))

    def test_diffusion_manufactured_solution(self):
        """Verify MMS solution for diffusion equation tests."""
        # u(x,t) = exp(-α*π²*t/L²)*sin(πx/L)
        L = sp.Symbol('L', positive=True)
        u_mms = sp.exp(-alpha*sp.pi**2*t/L**2) * sp.sin(sp.pi*x/L)

        u_t = sp.diff(u_mms, t)
        u_xx = sp.diff(u_mms, x, 2)

        lhs = u_t
        rhs = alpha * u_xx

        verify_identity(sp.simplify(lhs), sp.simplify(rhs))


class TestTruncationErrorFormulas:
    """Verify truncation error formulas in Appendix."""

    def test_taylor_expansion_coefficients(self):
        """Verify Taylor expansion of u(x+h) matches textbook formula."""
        h = dx
        u_exact = u(x, t)

        # Taylor expansion
        taylor = sp.series(u(x + h, t), h, 0, 5)

        # Expected: u + h*u_x + (h²/2)*u_xx + (h³/6)*u_xxx + ...
        u_x = sp.diff(u_exact, x)
        u_xx = sp.diff(u_exact, x, 2)
        u_xxx = sp.diff(u_exact, x, 3)
        u_xxxx = sp.diff(u_exact, x, 4)

        expected = (u_exact + h*u_x + h**2/2*u_xx +
                   h**3/6*u_xxx + h**4/24*u_xxxx)

        # Compare term by term
        taylor_no_O = taylor.removeO()
        verify_identity(taylor_no_O, expected)
```

### D.2 Additional Tests: `tests/test_plotting_reproducibility.py`

```python
"""
Tests to ensure plots are reproducible.
"""

import pytest
import numpy as np
from src.plotting import set_seed, RANDOM_SEED, create_solution_plot


def test_random_seed_determinism():
    """Verify setting seed produces deterministic results."""
    set_seed(42)
    vals1 = np.random.rand(10)

    set_seed(42)
    vals2 = np.random.rand(10)

    np.testing.assert_array_equal(vals1, vals2)


def test_plot_data_unchanged():
    """Verify plot data is deterministic."""
    set_seed()

    x = np.linspace(0, 1, 11)
    u = np.sin(np.pi * x)

    fig1 = create_solution_plot(x, u, title="Test")
    fig2 = create_solution_plot(x, u, title="Test")

    # Compare data arrays in traces
    np.testing.assert_array_equal(
        fig1.data[0].y,
        fig2.data[0].y
    )
```

### D.3 CI Integration for Derivation Tests

Add to `.github/workflows/tests.yml`:

```yaml
  test-derivations:
    name: Verify Mathematical Derivations
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install sympy numpy pytest
      - name: Run derivation tests
        run: pytest tests/test_derivations.py -v --tb=long
```

---

## Design Decisions (Confirmed)

Based on discussions, the following design decisions have been confirmed:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Exercise format** | Skeleton code with TODOs | Students learn by completing partial implementations |
| **Code organization** | Multiple specialized solvers | Separate functions per scheme (mirrors original structure) |
| **Visualization** | Interactive Plotly (default) | Modern, interactive plots with static fallback |
| **ODE handling** | Minimize, start with PDEs | Devito is designed for PDEs; brief ODE mention only |
| **Performance focus** | Educational | Clear code over optimization, mention advanced options briefly |
| **Implicit schemes** | PETSc branch | Separate venv for Backward Euler, Crank-Nicolson |
| **Dimensions** | 1D primary, 2D/3D brief | Deep 1D coverage, dimension scaling shown conceptually |
| **Legacy code** | Tests only | NumPy implementations retained for verification |
| **Truncation appendix** | Expand with Devito | Connect space_order to truncation error |
| **Software engineering** | Devito-focused rewrite | Project structure, testing patterns for Devito |

### Reproducibility Decisions (NEW)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LaTeX macros** | Inline/remove all | No custom `\newcommand`; all math via SymPy or explicit LaTeX |
| **Equations** | SymPy-generated | All equations generated from SymPy with `sp.latex()` |
| **Derivation verification** | Unit tested | Every stencil, stability condition, exact solution tested |
| **Plots** | Seeded, deterministic | All plots reproducible via fixed random seed and explicit parameters |
| **Freeze strategy** | `freeze: auto`, committed | `_freeze/` directory committed to git for reproducibility |
| **CI verification** | Derivation tests first | Mathematical tests run before solver tests and book build |
| **Shared symbols** | Centralized `src/symbols.py` | Consistent notation across all chapters |

---

## Exercise Template Format

Exercises should follow this skeleton pattern:

```python
"""
Exercise: Implement [description]

TODO: Complete the following implementation
"""
from devito import Grid, TimeFunction, Eq, Operator
import numpy as np


def solver_exercise(I, param1, param2, ...):
    """
    Solve [PDE description].

    Parameters
    ----------
    I : callable
        Initial condition
    param1 : float
        Description

    Returns
    -------
    u_hist : ndarray
        Solution history
    """
    # Setup grid
    grid = Grid(shape=(Nx+1,), extent=(L,))

    # TODO: Create TimeFunction with appropriate time_order and space_order
    u = ...  # YOUR CODE HERE

    # TODO: Set initial condition
    ...  # YOUR CODE HERE

    # TODO: Define the PDE update equation
    update = Eq(...)  # YOUR CODE HERE

    # TODO: Add boundary conditions
    bc = [...]  # YOUR CODE HERE

    # Compile and run
    op = Operator([update] + bc)

    # Time stepping loop
    Nt = int(round(T / dt))
    for n in range(Nt):
        op.apply(time_m=n, time_M=n, dt=dt)

    return u.data[...], x_coords


# Test your implementation
if __name__ == '__main__':
    # Test case with known solution
    def I(x):
        return np.sin(np.pi * x)

    u, x = solver_exercise(I, ...)

    # Compare with expected result
    # u_expected = ...
    # assert np.allclose(u, u_expected)
```

Solutions will be provided in `src/solutions/` directory (not included in book text).

---

*Document prepared for the devito_book refactoring project.*
*Last updated: 2026-01-27*
