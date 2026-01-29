# Contributing to *Finite Difference Computing with PDEs*

Thank you for your interest in contributing to this book. This guide outlines the practices and standards we follow to maintain quality, reproducibility, and consistency.

## Table of Contents

- [Pedagogical Philosophy and Priorities](#pedagogical-philosophy-and-priorities)
- [Content Gaps and Contribution Opportunities](#content-gaps-and-contribution-opportunities)
- [Development Workflow](#development-workflow)
- [Mathematical Derivations with SymPy](#mathematical-derivations-with-sympy)
- [Finite Difference Code with Devito](#finite-difference-code-with-devito)
- [Code Verification](#code-verification)
- [Plotting and Visualization](#plotting-and-visualization)
- [Image Generation](#image-generation)
- [Quarto Document Guidelines](#quarto-document-guidelines)
- [Testing](#testing)
- [Linting and Code Quality](#linting-and-code-quality)
- [Local Development](#local-development)
- [Attribution and Licensing](#attribution-and-licensing)

---

## Pedagogical Philosophy and Priorities

This book aims to teach finite difference methods with a modern, rigorous, and reproducible approach. Contributors should adhere to these pedagogical principles:

### 1. Theory Before Code

Every numerical method must be grounded in theory:

- **Derive before implementing**: Start with Taylor series derivations showing how FD formulas arise
- **Analyze before running**: Present truncation error, consistency, and stability analysis
- **Verify before trusting**: Show convergence rates match theoretical predictions

Example progression for a new scheme:
1. Mathematical derivation (Taylor expansion, truncation error)
2. Stability analysis (Von Neumann / Fourier analysis)
3. SymPy verification of the derivation
4. Devito implementation
5. Numerical verification (MMS, convergence rates)
6. Physical application with interpretation

### 2. Lax Equivalence as Foundation

The Lax Equivalence Theorem underpins all FD analysis:

> *For a consistent finite difference method, stability is equivalent to convergence.*

Every scheme presentation should address:
- **Consistency**: Does the scheme approximate the PDE as h → 0?
- **Stability**: Do errors remain bounded? (CFL conditions, Von Neumann analysis)
- **Convergence**: What is the order of accuracy? (Verify empirically)

### 3. Error Analysis Throughout

Quantitative error analysis is mandatory, not optional:

- **Truncation error**: Derive symbolically using SymPy
- **Dispersion error**: For wave problems, analyze phase velocity
- **Dissipation error**: For diffusion/advection, analyze amplitude decay
- **Rounding error**: Discuss precision effects for intensive computations

### 4. Progressive Complexity

Content should build systematically:

```
1D scalar → 2D scalar → Systems → Nonlinear → Coupled physics
Explicit → Implicit → High-order → Structure-preserving
Cartesian → Curvilinear → Complex geometry
Forward problem → Inverse problem → Optimization
```

### 5. Motivating Applications

Every method needs compelling applications:

| PDE Type | Physical Applications |
|----------|----------------------|
| Elliptic | Steady heat, electrostatics, structural stress |
| Parabolic | Transient heat, diffusion, Black-Scholes pricing |
| Hyperbolic | Waves, acoustics, seismology, traffic flow |
| Mixed | Advection-diffusion, Navier-Stokes, reaction-diffusion |

### 6. Reproducibility as Requirement

All results must be reproducible:

- Fixed random seeds (`set_seed(42)`)
- Version-pinned dependencies
- Generator scripts for all figures
- Automated tests validating examples

---

## Content Gaps and Contribution Opportunities

The following topics have been identified as gaps in the current book. Contributors are encouraged to address these areas following the pedagogical principles above.

### High Priority Gaps

#### 1. Rigorous Numerical Analysis Foundations

**Current state**: Truncation errors covered; formal stability/convergence theory incomplete.

**Needed**:
- Dedicated chapter on Lax Equivalence Theorem with proofs
- Systematic Von Neumann stability analysis for all schemes
- Norm-based error analysis (L1, L2, L∞)
- CFL condition derivations from first principles

**Contribution guidance**:
```python
# Use SymPy to derive stability conditions
from src.symbols import dt, dx, C, F
import sympy as sp

# Von Neumann analysis: substitute u^n_j = G^n * exp(i*k*j*dx)
G = sp.Symbol('G')  # Amplification factor
k = sp.Symbol('k', real=True)  # Wavenumber

# For FTCS diffusion: G = 1 - 4F*sin^2(k*dx/2)
# Stability requires |G| <= 1
```

#### 2. Elliptic PDEs and Linear Solvers

**Current state**: Not covered.

**Needed**:
- Chapter on Poisson/Laplace equations
- 5-point and 9-point Laplacian stencils
- Direct solvers (banded LU, sparse factorization)
- Iterative solvers (Jacobi, Gauss-Seidel, SOR, Conjugate Gradient)
- Multigrid methods (V-cycle, W-cycle)

**Devito approach**: Use `Function` (not `TimeFunction`) and external linear solvers:
```python
from devito import Grid, Function, Eq, Operator
from scipy.sparse.linalg import spsolve

grid = Grid(shape=(nx, ny), extent=(Lx, Ly))
u = Function(name='u', grid=grid, space_order=2)

# Devito for stencil definition, scipy for solve
laplacian = u.dx2 + u.dy2
# Extract sparse matrix, solve with spsolve or multigrid
```

#### 3. High-Order Finite Difference Schemes

**Current state**: Only second-order schemes.

**Needed**:
- Fourth and sixth-order central differences
- Compact (Padé) schemes with spectral-like resolution
- Dispersion-Relation-Preserving (DRP) schemes
- Comparison of dispersion relations across orders

**Key reference**: Lele (1992) compact schemes paper.

**SymPy derivation example**:
```python
from src.operators import taylor_expand
import sympy as sp

# Derive 4th-order central difference for u_xx
# Stencil: (−u_{i-2} + 16u_{i-1} − 30u_i + 16u_{i+1} − u_{i+2}) / (12 dx^2)
coeffs = sp.Rational(-1, 12), sp.Rational(16, 12), sp.Rational(-30, 12), ...
```

#### 4. Dispersion and Dissipation Analysis

**Current state**: Mentioned briefly; no systematic treatment.

**Needed**:
- Numerical dispersion relations for wave schemes
- Phase velocity error plots
- Group velocity analysis
- Comparison: upwind (dissipative) vs. central (dispersive)

**Verification approach**:
```python
# Compare numerical vs analytical dispersion relation
# Analytical: omega = c * k
# Numerical: omega_h = (1/dt) * arcsin(C * sin(k*dx))
```

#### 5. Structure-Preserving Methods (SBP/Mimetic)

**Current state**: Not covered.

**Needed**:
- Summation-By-Parts (SBP) operators
- Discrete energy stability proofs
- Simultaneous Approximation Terms (SAT) for boundaries
- Energy-conserving schemes for wave equations

**Why it matters**: Long-time simulations require schemes that preserve physical invariants.

#### 6. Curvilinear and Mapped Grids

**Current state**: Cartesian grids only.

**Needed**:
- Coordinate transformation theory
- Metric terms and Jacobians
- Geometric Conservation Law (GCL)
- Examples: polar, cylindrical, body-fitted grids

**Devito note**: Devito supports subdomains but not general curvilinear coordinates natively. Contributions may need to implement metric terms explicitly.

#### 7. Adjoint-State Methods and Inverse Problems

**Current state**: Not covered (though Devito supports adjoints).

**Needed**:
- Adjoint PDE derivation from Lagrangian
- Discrete vs. continuous adjoints
- Gradient computation for optimization
- Full Waveform Inversion (FWI) as case study

**Devito strength**: Devito can automatically generate adjoint operators:
```python
from devito import Operator

forward_op = Operator([forward_eq])
# Devito can derive adjoint via .adj_derivative or manual construction
```

### Medium Priority Gaps

#### 8. ENO/WENO Schemes for Shocks

**Current state**: Not covered.

**Needed**:
- Essentially Non-Oscillatory (ENO) reconstruction
- Weighted ENO (WENO) schemes
- Shock-capturing without oscillations
- Applications: Burgers' equation shocks, gas dynamics

**Key reference**: Shu (1998) ENO/WENO lecture notes.

#### 9. Implicit Time Stepping and ADI

**Current state**: Mentioned but not systematically covered.

**Needed**:
- Backward Euler implementation details
- Crank-Nicolson stability analysis
- Alternating Direction Implicit (ADI) for 2D/3D
- When to use implicit vs. explicit

#### 10. Mixed Precision and Performance

**Current state**: Not covered.

**Needed**:
- Floating-point error analysis
- When FP32/FP16 is acceptable
- GPU acceleration patterns
- Memory bandwidth optimization

**Devito note**: Devito handles much of this automatically, but understanding the tradeoffs is valuable.

### Lower Priority (Advanced Topics)

#### 11. Systems of PDEs

- Maxwell's equations (FDTD/Yee scheme)
- Elastic wave equations
- Shallow water equations
- Navier-Stokes (simplified)

#### 12. Multi-Physics Coupling

- Reaction-diffusion systems
- Thermo-mechanical coupling
- Fluid-structure interaction basics

---

### Chapter Template for New Content

When adding a new chapter or major section, follow this structure:

```markdown
# Chapter Title {#sec-chapter-id}

## Introduction and Motivation
- Physical problem and governing PDE
- Why this method/topic matters
- Learning objectives

## Mathematical Derivation
- Taylor series / variational derivation
- Truncation error analysis (with SymPy)
- Stability analysis

## Devito Implementation
- Complete solver following standard pattern
- Step-by-step explanation
- Boundary condition handling

## Verification
- Exact solution test (if available)
- Method of Manufactured Solutions
- Convergence rate verification
- Conservation property checks

## Applications
- Physical example with realistic parameters
- Visualization and interpretation
- Comparison with analytical/reference solutions

## Exercises
- Derivation exercises (pen and paper + SymPy)
- Implementation exercises (extend the solver)
- Analysis exercises (stability, convergence)

## Summary
- Key takeaways
- Connection to other chapters
- Further reading
```

---

## Development Workflow

### Branch-Based Development

All contributions must follow this workflow:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following the guidelines in this document.

3. **Run tests locally** before pushing:
   ```bash
   pytest tests/ -v
   quarto render --to pdf
   ```

4. **Push and create a Pull Request** to `main`:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. **Wait for CI checks** - all GitHub Actions must pass:
   - `test-derivations` - SymPy/mathematical verification
   - `test-devito-explicit` - Devito solver tests
   - `lint` - Code quality checks
   - `build-book` - Quarto PDF/HTML build

6. **Code review required** - at least one reviewer must approve before merging.

7. **Merge to main** - triggers automatic publication to GitHub Pages.

### Commit Messages

Follow conventional commit style:
```
type: Short description (max 70 chars)

Longer description if needed, explaining the "why" not the "what".
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

---

## Mathematical Derivations with SymPy

### Why SymPy?

We use SymPy to:
1. **Verify correctness** of mathematical derivations programmatically
2. **Generate LaTeX** consistently rather than writing it by hand
3. **Ensure reproducibility** of all formulas in the book

### Standard Symbols

Always use symbols from `src/symbols.py` for consistency:

```python
from src.symbols import (
    # Spatial variables
    x, y, z,
    # Grid spacing (positive, with LaTeX notation)
    dx, dy, dz, dt,
    # Dimensionless numbers
    C,      # Courant number
    F,      # Fourier number
    # Physical parameters
    c,      # Wave speed
    alpha,  # Diffusivity
    nu,     # Viscosity
    # Index symbols (integers)
    i, j, n, m,
    # Helper functions
    half, third, quarter,
)
```

### Finite Difference Operators

Use operators from `src/operators.py`:

```python
from src.operators import (
    forward_diff,
    backward_diff,
    central_diff,
    second_derivative_central,
    laplacian_2d,
    taylor_expand,
    derive_truncation_error,
)
import sympy as sp

# Example: Derive central difference formula
u = sp.Function('u')
stencil = central_diff(u(x), x, dx)

# Get LaTeX for the book
latex_expr = sp.latex(stencil)
print(latex_expr)  # \frac{u{\left(dx + x \right)} - u{\left(- dx + x \right)}}{2 dx}

# Verify truncation error order
error = derive_truncation_error(stencil, u(x).diff(x), x, dx, order=4)
print(f"Truncation error: {sp.latex(error)}")
```

### Display Utilities for Quarto

Use `src/display.py` for Quarto-compatible output:

```python
from src.display import show_eq, show_eq_aligned, latex_expr

# Single equation with label
show_eq(u.dt2, c**2 * u.dx2, label='eq-wave-1d')
# Output: $$ u_{tt} = c^2 u_{xx} $$ {#eq-wave-1d}

# Multi-line aligned equations
show_eq_aligned([
    (u.forward, "2u^n - u^{n-1} + C^2(u_{i+1}^n - 2u_i^n + u_{i-1}^n)"),
], label='eq-wave-stencil')
```

### Verification of Derivations

All mathematical derivations should have corresponding tests:

```python
# In tests/test_derivations.py
def test_central_diff_is_second_order():
    """Verify central difference has O(h^2) truncation error."""
    from src.operators import central_diff, get_stencil_order

    stencil = central_diff(u(x), x, h)
    exact = sp.Derivative(u(x), x)
    order = get_stencil_order(stencil, exact, x, h)

    assert order == 2, f"Expected O(h^2), got O(h^{order})"
```

---

## Finite Difference Code with Devito

### Why Devito?

[Devito](https://www.devitoproject.org/) is a domain-specific language for:
- **Symbolic PDE specification** - write equations mathematically
- **Automatic code generation** - generates optimized C code
- **Performance portability** - runs on CPUs, GPUs, clusters

### Standard Solver Structure

All Devito solvers must follow this pattern:

```python
"""
Module: solve_pde_1d.py

Solves the 1D PDE: u_t = alpha * u_xx

Domain: x in [0, L]
Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
Initial condition: u(x,0) = I(x)

Discretization:
- Time: Forward Euler, O(dt)
- Space: Central difference, O(dx^2)

Update formula:
    u^{n+1}_i = u^n_i + F(u^n_{i-1} - 2u^n_i + u^n_{i+1})

where F = alpha * dt / dx^2 (Fourier number, F <= 0.5 for stability)
"""
from dataclasses import dataclass
import numpy as np

try:
    from devito import Grid, TimeFunction, Eq, Operator, Constant, solve
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


@dataclass
class DiffusionResult:
    """Container for diffusion solver results.

    Attributes:
        u: Final solution array
        x: Spatial coordinate array
        t: Final time
        dt: Time step used
        F: Fourier number
        u_history: Optional time history of solution
        t_history: Optional array of time values
    """
    u: np.ndarray
    x: np.ndarray
    t: float
    dt: float
    F: float
    u_history: np.ndarray | None = None
    t_history: np.ndarray | None = None


def solve_diffusion_1d(
    L: float,
    alpha: float,
    Nx: int,
    T: float,
    I: callable,
    F: float = 0.5,
    store_history: bool = False,
) -> DiffusionResult:
    """Solve the 1D diffusion equation using Devito.

    Parameters
    ----------
    L : float
        Domain length [0, L]
    alpha : float
        Diffusivity coefficient
    Nx : int
        Number of spatial grid points
    T : float
        Final simulation time
    I : callable
        Initial condition function I(x)
    F : float, optional
        Fourier number (default 0.5, maximum for stability)
    store_history : bool, optional
        Whether to store solution at all time steps

    Returns
    -------
    DiffusionResult
        Dataclass containing solution and metadata

    Raises
    ------
    ImportError
        If Devito is not installed
    ValueError
        If F > 0.5 (unstable)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito required. Install with: pip install devito")

    if F > 0.5:
        raise ValueError(f"Fourier number F={F} > 0.5 violates stability condition")

    # Compute grid spacing and time step
    dx = L / Nx
    dt = F * dx**2 / alpha
    Nt = int(np.ceil(T / dt))

    # Create Devito grid and field
    grid = Grid(shape=(Nx + 1,), extent=(L,))
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

    # Initialize
    x_coords = np.linspace(0, L, Nx + 1)
    u.data[0, :] = I(x_coords)
    u.data[1, :] = I(x_coords)

    # Symbolic PDE: u_t = alpha * u_xx
    alpha_const = Constant(name='alpha')
    pde = u.dt - alpha_const * u.dx2
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Boundary conditions
    t_dim = grid.stepping_dim
    bc_left = Eq(u[t_dim + 1, 0], 0)
    bc_right = Eq(u[t_dim + 1, Nx], 0)

    # Create operator
    op = Operator([stencil, bc_left, bc_right])

    # Time stepping with optional history storage
    if store_history:
        u_history = np.zeros((Nt + 1, Nx + 1))
        t_history = np.zeros(Nt + 1)
        u_history[0, :] = u.data[0, :]
        t_history[0] = 0

        for n in range(Nt):
            op.apply(time_m=0, time_M=0, dt=dt, alpha=alpha)
            u_history[n + 1, :] = u.data[0, :]
            t_history[n + 1] = (n + 1) * dt
    else:
        op.apply(time_m=0, time_M=Nt - 1, dt=dt, alpha=alpha)
        u_history = None
        t_history = None

    return DiffusionResult(
        u=u.data[0, :].copy(),
        x=x_coords,
        t=Nt * dt,
        dt=dt,
        F=F,
        u_history=u_history,
        t_history=t_history,
    )
```

### Key Devito Concepts

| Concept | Usage | Purpose |
|---------|-------|---------|
| `Grid` | `Grid(shape=(nx,), extent=(L,))` | Define computational domain |
| `TimeFunction` | `TimeFunction(name='u', grid=grid, time_order=2, space_order=2)` | Time-dependent field |
| `Function` | `Function(name='c', grid=grid)` | Spatially-varying coefficient |
| `Constant` | `Constant(name='alpha')` | Runtime parameter |
| `Eq` | `Eq(u.forward, rhs)` | Symbolic equation |
| `solve` | `solve(pde, u.forward)` | Isolate unknown |
| `Operator` | `Operator([eq1, eq2, ...])` | Compile to C |
| Derivatives | `u.dt`, `u.dt2`, `u.dx`, `u.dx2` | Symbolic derivatives |

### Devito Solver Checklist

- [ ] Module docstring with PDE, domain, BCs, discretization, update formula
- [ ] Dataclass for results with type hints
- [ ] Comprehensive function docstring (NumPy style)
- [ ] `DEVITO_AVAILABLE` guard with helpful error message
- [ ] Stability condition check with informative error
- [ ] Clear variable naming matching mathematical notation
- [ ] Comments explaining each step
- [ ] Support for optional history storage
- [ ] Return copy of data (Devito arrays are mutable)

---

## Code Verification

### Verification Hierarchy

Every solver must have verification at multiple levels:

#### 1. Exact Solution Reproduction

Test with solutions where the numerical scheme is exact:

```python
def test_quadratic_exact():
    """Quadratic polynomial should be exact for O(dx^2) scheme."""
    def I(x):
        return x * (1 - x)  # Quadratic

    result = solve_diffusion_1d(L=1.0, alpha=1.0, Nx=10, T=0.1, I=I)

    # Analytical solution for u_t = u_xx with u(x,0) = x(1-x)
    def exact(x, t):
        return x * (1 - x) - 2 * t

    expected = exact(result.x, result.t)
    np.testing.assert_allclose(result.u, expected, rtol=1e-10)
```

#### 2. Method of Manufactured Solutions (MMS)

For complex PDEs, manufacture a solution and verify convergence:

```python
def test_mms_convergence():
    """Verify expected convergence rate using MMS."""
    from src.verification import manufactured_solution, convergence_rate

    # Manufactured solution: u_e = sin(pi*x) * exp(-t)
    u_exact = lambda x, t: np.sin(np.pi * x) * np.exp(-t)

    errors = []
    dx_values = [0.1, 0.05, 0.025, 0.0125]

    for dx in dx_values:
        Nx = int(1.0 / dx)
        result = solve_diffusion_1d(L=1.0, alpha=1.0, Nx=Nx, T=0.1, I=lambda x: np.sin(np.pi * x))
        error = np.max(np.abs(result.u - u_exact(result.x, result.t)))
        errors.append(error)

    rate = convergence_rate(dx_values, errors)
    assert rate > 1.9, f"Expected O(dx^2), got rate {rate:.2f}"
```

#### 3. Conservation Laws

Verify physical invariants:

```python
def test_energy_conservation():
    """Wave equation should conserve total energy."""
    result = solve_wave_1d(L=1.0, c=1.0, Nx=100, T=1.0, store_history=True)

    def energy(u, v, dx):
        return 0.5 * dx * np.sum(u**2 + v**2)

    E0 = energy(result.u_history[0], np.zeros_like(result.u_history[0]), result.dx)
    E_final = energy(result.u_history[-1], ..., result.dx)

    np.testing.assert_allclose(E_final, E0, rtol=1e-6)
```

#### 4. Boundary Condition Verification

```python
def test_dirichlet_bc():
    """Verify boundary conditions are satisfied."""
    result = solve_wave_1d(L=1.0, c=1.0, Nx=50, T=0.5, store_history=True)

    # u(0, t) = 0 and u(L, t) = 0 for all t
    np.testing.assert_allclose(result.u_history[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(result.u_history[:, -1], 0, atol=1e-12)
```

---

## Plotting and Visualization

### Standard Configuration

Use `src/plotting.py` for consistent, reproducible plots:

```python
from src.plotting import (
    set_seed,
    get_color_scheme,
    create_solution_plot,
    create_convergence_plot,
    COLORS,
)

# Always set seed for reproducibility
set_seed(42)

# Use colorblind-friendly palette
colors = get_color_scheme('accessible')
```

### Color Palette

Use semantic colors from `COLORS`:

```python
COLORS = {
    'numerical': '#1f77b4',    # Blue - computed solution
    'exact': '#ff7f0e',        # Orange - analytical solution
    'error': '#d62728',        # Red - error plots
    'initial': '#2ca02c',      # Green - initial conditions
    'boundary': '#9467bd',     # Purple - boundary-related
}
```

### Plot Style Guidelines

1. **Labels**: Always include axis labels with units
2. **Legends**: Place outside plot area if possible
3. **Grid**: Use light grid for readability
4. **Font size**: Minimum 10pt for PDF legibility
5. **Figure size**: (8, 6) inches default for single plots
6. **DPI**: 300 for final figures, 100 for previews

### Example

```python
import matplotlib.pyplot as plt
from src.plotting import COLORS, set_seed

set_seed(42)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, u_numerical, '-', color=COLORS['numerical'], label='Numerical', linewidth=1.5)
ax.plot(x, u_exact, '--', color=COLORS['exact'], label='Exact', linewidth=1.5)
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$u(x, t)$', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('fig/solution.pdf', dpi=300, bbox_inches='tight')
```

---

## Image Generation

### Reproducibility Requirements

1. **Generator scripts**: Every figure should have an associated Python script
2. **Random seeds**: Always use `set_seed(42)` at script start
3. **Committed to repo**: Both script and generated images
4. **Multiple formats**: Generate `.pdf` (print) and `.png` (web)

### Directory Structure

```
chapters/wave/
├── fig/
│   ├── standing_wave.py      # Generator script
│   ├── standing_wave.pdf     # PDF for print
│   ├── standing_wave.png     # PNG for web
│   └── README.md             # Documents how to regenerate
```

### Generator Script Template

```python
#!/usr/bin/env python3
"""Generate standing wave figure for Chapter 2.

Usage:
    python standing_wave.py

Outputs:
    standing_wave.pdf
    standing_wave.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reproducibility
np.random.seed(42)

# Figure generation code
...

# Save in multiple formats
output_dir = Path(__file__).parent
fig.savefig(output_dir / 'standing_wave.pdf', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'standing_wave.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    print(f"Generated: standing_wave.pdf, standing_wave.png")
```

### Quarto Figure Syntax

```markdown
![Standing wave modes for the 1D wave equation](fig/standing_wave.pdf){#fig-standing-wave width="80%"}

Reference with @fig-standing-wave.
```

---

## Quarto Document Guidelines

### File Structure

Each chapter follows this structure:

```
chapters/wave/
├── index.qmd           # Chapter introduction
├── wave1D_fd1.qmd      # Section files
├── wave1D_fd2.qmd
├── wave1D_prog.qmd
└── fig/                # Chapter figures
```

### Cross-References

Use Quarto's cross-reference system:

| Type | Label Syntax | Reference Syntax |
|------|--------------|------------------|
| Section | `{#sec-wave-1d}` | `@sec-wave-1d` |
| Equation | `{#eq-wave-update}` | `@eq-wave-update` |
| Figure | `{#fig-standing}` | `@fig-standing` |
| Table | `{#tbl-errors}` | `@tbl-errors` |

### Equation Labeling

**Single equation:**
```markdown
$$
u_{tt} = c^2 u_{xx}
$$ {#eq-wave-1d}
```

**Aligned equations (use pure LaTeX for individual line labels):**
```latex
\begin{align}
u^{n+1}_i &= 2u^n_i - u^{n-1}_i + C^2(u^n_{i+1} - 2u^n_i + u^n_{i-1}) \label{eq:wave-update} \\
C &= \frac{c \Delta t}{\Delta x} \label{eq:courant}
\end{align}
```

### Code Blocks

**Python code (displayed, not executed):**
````markdown
```python
from devito import Grid, TimeFunction

grid = Grid(shape=(100,), extent=(1.0,))
u = TimeFunction(name='u', grid=grid)
```
````

**Include from file:**
````markdown
```{.python include="../../src/wave/wave1D_devito.py" start-line=50 end-line=75}
```
````

### Custom LaTeX Macros

Use macros defined in `_quarto.yml` for consistency:

| Macro | Renders as | Usage |
|-------|------------|-------|
| `\half` | 1/2 | Fractions |
| `\tp` | thin period | End of equations |
| `\uex` | u_e | Exact solution |
| `\Oof{x}` | O(x) | Order notation |
| `\dfc` | alpha | Diffusion coefficient |

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Skip Devito tests (faster, math only)
pytest tests/ -v -m "not devito"

# Devito tests only
pytest tests/ -v -m devito

# Mathematical derivation tests only
pytest tests/ -v -m derivation

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Organization

```
tests/
├── conftest.py              # Fixtures and configuration
├── test_operators.py        # SymPy operator verification
├── test_derivations.py      # Mathematical identity verification
├── test_wave_devito.py      # Wave equation solvers
├── test_diffu_devito.py     # Diffusion solvers
├── test_advec_devito.py     # Advection solvers
└── test_nonlin_devito.py    # Nonlinear solvers
```

### Pytest Markers

```python
@pytest.mark.slow           # Long-running tests
@pytest.mark.devito         # Requires Devito installation
@pytest.mark.derivation     # Mathematical verification
```

### Writing Tests

```python
import pytest
import numpy as np

@pytest.mark.devito
class TestWave1DSolver:
    """Tests for 1D wave equation solver."""

    def test_stability_check(self):
        """Solver should reject unstable Courant numbers."""
        with pytest.raises(ValueError, match="Courant number"):
            solve_wave_1d(L=1.0, c=1.0, Nx=10, T=1.0, C=1.5)

    def test_convergence_rate(self):
        """Verify O(dt^2 + dx^2) convergence."""
        # Implementation...
        assert rate > 1.9
```

---

## Linting and Code Quality

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit:

```bash
# Install hooks (one time)
pip install -e ".[dev]"
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run with auto-fix (manual stage)
pre-commit run --hook-stage manual --all-files
```

### Automatic Checks

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing spaces |
| `end-of-file-fixer` | Ensure newline at EOF |
| `check-yaml` | Validate YAML syntax |
| `check-added-large-files` | Block files >500KB |
| `isort` | Import ordering |
| `ruff` | Python linting |
| `typos` | Spell checking |
| `markdownlint-cli2` | Markdown formatting |

### Ruff Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 90
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "B", "UP", "SIM"]
```

### Import Ordering

isort organizes imports in this order:
1. Standard library
2. Third-party packages
3. Local imports

```python
# Correct
import os
from pathlib import Path

import numpy as np
import sympy as sp
from devito import Grid, TimeFunction

from src.symbols import x, dx
from src.operators import central_diff
```

---

## Local Development

### Environment Setup

```bash
# Clone repository
git clone https://github.com/devitocodes/devito_book.git
cd devito_book

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Install Devito (optional, for solver development)
pip install -e ".[devito]"

# Install pre-commit hooks
pre-commit install
```

### Building the Book

```bash
# Install Quarto (see https://quarto.org/docs/get-started/)
# macOS: brew install quarto
# Linux: see Quarto website

# Install TinyTeX for PDF generation
quarto install tinytex

# Build PDF only
quarto render --to pdf

# Build all formats (HTML + PDF)
quarto render

# Live preview with hot reload
quarto preview
```

### Output Location

- PDF: `_book/Finite-Difference-Computing-with-PDEs.pdf`
- HTML: `_book/index.html`

### Common Issues

**TinyTeX missing packages:**
```bash
quarto install tinytex --update-path
tlmgr install <package-name>
```

**Devito installation issues:**
```bash
# Try installing from source
pip install git+https://github.com/devitocodes/devito.git
```

---

## Attribution and Licensing

### License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Required Attribution

When reusing content from this book, you must credit:

1. **Original authors**: Hans Petter Langtangen and Svein Linge
2. **Original work**: *Finite Difference Computing with PDEs: A Modern Software Approach* (Springer, 2017)
3. **Original DOI**: <https://doi.org/10.1007/978-3-319-55456-3>
4. **Adapter**: Gerard J. Gorman
5. **Changes**: Modernized with Devito DSL, Quarto, and modern Python practices

### Third-Party Content

When adding content from external sources:

1. **Verify license compatibility** with CC BY 4.0
2. **Document attribution** in both:
   - The relevant chapter file (footnote or acknowledgment)
   - `references.bib` with full citation
3. **Obtain explicit permission** for content not under open license
4. **Keep records** of permissions in project documentation

### Code Attribution

When adapting code from external sources:

```python
# Adapted from [Source Name] by [Author]
# Original: [URL]
# License: [License Name]
# Modifications: [Brief description of changes]
```

### Image Attribution

For third-party images:

```markdown
![Description](fig/image.png){#fig-label}

*Source: [Author/Organization], [License]. Used with permission.*
```

---

## Summary Checklist

Before submitting a pull request, verify:

### Code Quality
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Linting passes: `pre-commit run --all-files`
- [ ] New code has tests with >80% coverage
- [ ] Docstrings follow NumPy style

### Mathematical Content
- [ ] Derivations use SymPy from `src/symbols.py` and `src/operators.py`
- [ ] LaTeX generated programmatically where possible
- [ ] Truncation errors verified symbolically
- [ ] Tests in `test_derivations.py` for new formulas

### Devito Code
- [ ] Follows standard solver structure
- [ ] Module docstring with PDE, BCs, discretization
- [ ] Stability conditions checked with informative errors
- [ ] Results returned in dataclass
- [ ] Tests for exactness, convergence, conservation

### Documentation
- [ ] Quarto cross-references work: `@sec-`, `@eq-`, `@fig-`
- [ ] Images have generator scripts
- [ ] Third-party content properly attributed

### Workflow
- [ ] Branch created from latest `main`
- [ ] Commit messages follow convention
- [ ] CI checks pass
- [ ] Ready for code review

---

## Questions?

- Open an issue on [GitHub](https://github.com/devitocodes/devito_book/issues)
- See [Devito documentation](https://www.devitoproject.org/devito/index.html)
- See [Quarto documentation](https://quarto.org/docs/books/)
