# Implementation Roadmap: Extending the Devito Book

## Overview

Extend *Finite Difference Computing with PDEs* with content from `devito_repo/examples/`, prioritized by value and implementation effort.

### Guiding Principles

- **Low-hanging fruit first**: Leverage existing well-documented notebooks
- **Incremental value**: Each phase delivers a usable, complete chapter
- **No convenience classes**: All code uses explicit Devito API
- **Test-driven**: Tests required before content is considered complete

---

## Progress Summary

| Phase | Status | Tests | Commit |
|-------|--------|-------|--------|
| **Phase 1** | ✅ Complete | 62 | `ab9b38c0` |
| **Phase 2** | ✅ Complete | 73 | `b9e017b3` |
| **Phase 3** | ✅ Complete | 91 | - |
| **Phase 4** | 🔲 Not started | - | - |
| **Phase 5** | 🔲 Not started | - | - |
| **Phase 6** | 🔲 Not started | - | - |
| **Phase 7** | 🔲 Not started | - | - |

**Total tests: 411**

---

## Phase 1: Quick Wins (Existing CFD Content) ✅ COMPLETE

**Effort**: Low | **Value**: High | **Source**: Ready-to-use notebooks

These chapters have complete source material in `devito_repo/examples/cfd/` and fill obvious gaps in the current Part I.

### 1.1 Chapter 6: Elliptic PDEs and Iterative Solvers ✅

**Source**: `cfd/05_laplace.ipynb`, `cfd/06_poisson.ipynb`

**Why first**: Natural progression from time-dependent PDEs to steady-state. Uses `Function` instead of `TimeFunction` - an important Devito pattern not yet covered.

**Sections**:
- 6.1 Introduction to Elliptic PDEs (steady-state, BVPs)
- 6.2 The Laplace Equation (dual-buffer iteration pattern)
- 6.3 The Poisson Equation (source term handling)
- 6.4 Iterative Solver Analysis (Jacobi convergence)
- 6.5 Exercises

**Key Devito pattern**:
```python
p = Function(name='p', grid=grid, space_order=2)
pn = Function(name='pn', grid=grid, space_order=2)
# Dual-buffer Jacobi iteration with argument swapping
op(p=_p, pn=_pn)
```

**Deliverables**:
- [x] `chapters/elliptic/elliptic.qmd`
- [x] `src/elliptic/laplace_devito.py`
- [x] `src/elliptic/poisson_devito.py`
- [x] `tests/test_elliptic_devito.py` (18 tests)

---

### 1.2 Chapter 5 Enhancement: Burgers Equation ✅

**Source**: `cfd/04_burgers.ipynb`

**Why early**: Minimal addition to existing nonlinear chapter. Demonstrates `first_derivative()` with explicit order and shock formation.

**New sections for Chapter 5**:
- Burgers Equation (coupled 2D system)
- Mixed discretization (upwind advection, centered diffusion)

**Key Devito patterns**:
- `first_derivative()` with explicit `fd_order` and `side`
- Scalar and vector solver approaches

**Deliverables**:
- [x] `chapters/nonlin/burgers.qmd`
- [x] `src/nonlin/burgers_devito.py`
- [x] `tests/test_burgers_devito.py` (21 tests)

---

### 1.3 Chapter 7.1-7.2: Introduction to Systems + Shallow Water ✅

**Source**: `cfd/08_shallow_water_equation.ipynb`

**Why early**: First real PDE system (3 coupled equations). Introduces `ConditionalDimension` for snapshots naturally.

**Sections**:
- 7.1 Introduction to PDE Systems (conservation laws, coupling)
- 7.2 Shallow Water Equations (η, M, N system with bathymetry)

**Key Devito patterns**:
- Multiple coupled `TimeFunction` objects
- `Function` for bathymetry (static field)
- `ConditionalDimension` for output snapshots

**Deliverables**:
- [x] `chapters/systems/systems.qmd` (sections 7.1-7.2)
- [x] `src/systems/swe_devito.py`
- [x] `tests/test_swe_devito.py` (23 tests)

---

## Phase 2: High-Order Methods (Dispersion and DRP) ✅ COMPLETE

**Effort**: Medium | **Value**: High | **Source**: Ready notebooks

Essential for anyone doing wave propagation - explains why default stencils may not be enough.

### 2.1 Chapter 8: Dispersion Analysis and DRP Schemes ✅

**Source**: `seismic/tutorials/07.1_dispersion_relation.ipynb`, `seismic/tutorials/07_DRP_schemes.ipynb`

**Sections**:
- 8.1 Introduction to High-Order Methods
- 8.2 Dispersion Analysis (phase/group velocity, numerical dispersion)
- 8.3 The Fornberg Algorithm (computing FD weights)
- 8.4 Dispersion-Relation-Preserving (DRP) Schemes (Tam-Webb optimization)
- 8.5 Implementation in Devito (custom weights)
- 8.6 Comparison: Standard vs DRP Schemes
- 8.7 CFL Stability Condition
- 8.8 Exercises

**Key Devito pattern**:
```python
weights = np.array([...])  # DRP coefficients
u_lap = u.dx2(weights=weights) + u.dy2(weights=weights)
```

**Deliverables**:
- [x] `chapters/highorder/highorder.qmd`
- [x] `src/highorder/dispersion.py`
- [x] `src/highorder/drp_devito.py`
- [x] `tests/test_highorder_devito.py` (39 tests)

---

### 2.2 Chapter 7.3: Elastic Wave Equations ✅

**Source**: `seismic/tutorials/06_elastic.ipynb`, `seismic/tutorials/06_elastic_varying_parameters.ipynb`

**Why Phase 2**: Introduces `VectorTimeFunction`, `TensorTimeFunction`, and vector operators - foundational for later physics.

**Sections**:
- 7.3 Elastic Wave Equations
  - Velocity-stress formulation
  - VectorTimeFunction and TensorTimeFunction
  - Vector operators: div, grad, diag
  - Staggered grid discretization
  - Varying Lamé parameters

**Key Devito patterns**:
```python
from devito import VectorTimeFunction, TensorTimeFunction, div, grad, diag

v = VectorTimeFunction(name='v', grid=grid, time_order=1, space_order=so)
tau = TensorTimeFunction(name='tau', grid=grid, time_order=1, space_order=so, symmetric=True)

div_tau = div(tau)  # Divergence of tensor -> vector
grad_v = grad(v)    # Gradient of vector -> tensor
```

**Deliverables**:
- [x] Update `chapters/systems/systems.qmd` with section 7.3
- [x] `src/systems/elastic_devito.py`
- [x] `tests/test_elastic_devito.py` (34 tests)

---

## Phase 3: Advanced Schemes and Attenuation ✅ COMPLETE

**Effort**: Medium-High | **Value**: High | **Source**: Ready notebooks

### 3.1 Chapter 8.4-8.5: ADER and Staggered Grids ✅

**Source**: `16_ader_fd.ipynb`, `05_staggered_acoustic.ipynb`

**Sections**:
- 8.4 ADER Finite Difference Schemes (high-order time via spatial derivatives)
- 8.5 Staggered Grid Formulations (velocity-pressure systems)

**Key Devito pattern**:
```python
v = VectorTimeFunction(name='v', grid=grid, space_order=16, staggered=(None, None))
# ADER update with Taylor expansion in time
eq_p = Eq(p.forward, p + dt*pdt + (dt**2/2)*pdt2 + (dt**3/6)*pdt3 + (dt**4/24)*pdt4)
```

**Deliverables**:
- [x] Update `chapters/highorder/highorder.qmd` with sections 8.4-8.5
- [x] `src/highorder/ader_devito.py`
- [x] `src/highorder/staggered_devito.py`
- [x] `tests/test_ader_devito.py` (19 tests)
- [x] `tests/test_staggered_devito.py` (25 tests)

---

### 3.2 Chapter 7.4-7.5: Viscoacoustic and Viscoelastic Waves ✅

**Source**: `11_viscoacoustic.ipynb`, `09_viscoelastic.ipynb`

**Why here**: Builds on elastic waves, adds memory variables for attenuation.

**Sections**:
- 7.4 Viscoacoustic Waves (Q-attenuation, relaxation times)
- 7.5 Viscoelastic Waves (full 3D, multiple relaxation mechanisms)

**Key concepts**:
- Memory variables for dispersion
- Auxiliary equations for relaxation
- Three rheological models: SLS, Kelvin-Voigt, Maxwell

**Deliverables**:
- [x] Update `chapters/systems/systems.qmd` with sections 7.4-7.5
- [x] `src/systems/viscoacoustic_devito.py`
- [x] `src/systems/viscoelastic_devito.py`
- [x] `tests/test_viscoacoustic_devito.py` (24 tests)
- [x] `tests/test_viscoelastic_devito.py` (23 tests)

---

## Phase 4: Inverse Problems (Priority Content)

**Effort**: High | **Value**: Very High | **Source**: Ready notebooks

This is the stated priority - complete treatment of adjoint methods, RTM, and FWI.

### 4.1 Chapter 9: Inverse Problems and Optimization

**Source**: `02_rtm.ipynb`, `03_fwi.ipynb`, `13_LSRTM_acoustic.ipynb`, `seismic/inversion/fwi.py`

**Sections**:
- 9.1 Introduction to Inverse Problems
- 9.2 The Adjoint-State Method (SymPy derivation of Lagrangian)
- 9.3 Forward Modeling (full explicit code, no convenience classes)
- 9.4 Reverse Time Migration (RTM)
- 9.5 Adjoint Wavefield Computation
- 9.6 Gradient Computation
- 9.7 FWI Optimization Loop (scipy L-BFGS)
- 9.8 Regularization (Tikhonov, TV)
- 9.9 Least-Squares RTM (LSRTM)

**Critical**: Must rewrite all examples without `SeismicModel`, `AcousticWaveSolver`, etc.

**Key code pattern** (explicit API):
```python
# Manual Ricker wavelet
def ricker_wavelet(t, f0):
    t0 = 1.5 / f0
    return (1 - 2*(np.pi*f0*(t-t0))**2) * np.exp(-(np.pi*f0*(t-t0))**2)

# Explicit SparseTimeFunction
src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
src.coordinates.data[:] = [[500., 20.]]
src.data[:, 0] = ricker_wavelet(time_values, f0=10.)
```

**Deliverables**:
- [ ] `chapters/adjoint/adjoint.qmd`
- [ ] `src/adjoint/forward_devito.py`
- [ ] `src/adjoint/rtm_devito.py`
- [ ] `src/adjoint/fwi_devito.py`
- [ ] `src/adjoint/lsrtm_devito.py`
- [ ] `src/adjoint/gradient.py`
- [ ] `tests/test_adjoint_devito.py`
- [ ] `tests/test_rtm_devito.py`
- [ ] `tests/test_fwi_devito.py`
- [ ] `tests/test_lsrtm_devito.py`

---

## Phase 5: Performance and Scalability

**Effort**: Medium | **Value**: High | **Source**: Ready notebooks

Practical content for anyone running real simulations.

### 5.1 Chapter 15: GPU Computing

**Source**: `performance/01_gpu.ipynb`

- Devito GPU backends
- Memory management on GPU
- Performance comparison

### 5.2 Chapter 16: Memory Management and I/O

**Source**: `08_snapshotting.ipynb`, `12_time_blocking.ipynb`

**Sections**:
- 16.1 Wavefield Storage Strategies
- 16.2 Snapshotting with ConditionalDimension
- 16.3 Time Blocking and Compression

**Key pattern**:
```python
time_sub = ConditionalDimension('t_sub', parent=grid.time_dim, factor=10)
usave = TimeFunction(name='usave', grid=grid, save=nsnaps, time_dim=time_sub)
```

### 5.3 Chapter 17: Distributed Computing with Dask

**Source**: `04_dask.ipynb`

- Dask cluster integration
- Shot-parallel FWI
- Scaling to HPC

**Deliverables**:
- [ ] `chapters/performance/gpu.qmd`
- [ ] `chapters/performance/memory.qmd`
- [ ] `chapters/performance/distributed.qmd`
- [ ] `src/performance/snapshotting.py`
- [ ] `src/performance/checkpointing.py`
- [ ] Tests for performance patterns

---

## Phase 6: Domain Applications

**Effort**: Variable | **Value**: Medium-High | **Source**: Mixed

### 6.1 Quick Additions (Existing Notebooks)

**Chapter 10: Computational Finance**
- Source: `finance/bs_ivbp.ipynb`
- Black-Scholes PDE, non-standard SpaceDimension

**Chapter 11: Porous Media Flow**
- Source: `cfd/09_Darcy_flow_equation.ipynb`
- Darcy's law, permeability fields

**Chapter 12: CFD (Navier-Stokes)**
- Source: `cfd/07_cavity_flow.ipynb`
- Lid-driven cavity, projection method

### 6.2 New Development Required

**Chapter 13: Computational Electromagnetics (Maxwell)**
- Develop from scratch
- Yee grid / FDTD scheme
- E and H field staggering
- PML absorbing boundaries

**Chapter 14: Numerical Relativity**
- Develop from scratch
- ADM/BSSN formulation
- Gravitational wave extraction
- Single black hole example

---

## Phase 7: Theory Appendix

**Effort**: Low | **Value**: Medium

### Appendix D: Essential Numerical Analysis Theory

**Source**: `17_fourier_mode.ipynb` for D.4

- D.1 Lax Equivalence Theorem
- D.2 Von Neumann Stability Analysis
- D.3 Truncation Error Analysis (enhance existing)
- D.4 Fourier Mode Analysis (NEW)

---

## Summary: Implementation Order

| Phase | Content | Effort | Value | Status |
|-------|---------|--------|-------|--------|
| **1.1** | Elliptic PDEs | Low | High | ✅ Complete |
| **1.2** | Burgers | Low | Medium | ✅ Complete |
| **1.3** | Shallow Water | Low | High | ✅ Complete |
| **2.1** | Dispersion/DRP | Medium | High | ✅ Complete |
| **2.2** | Elastic Waves | Medium | High | ✅ Complete |
| **3.1** | ADER/Staggered | Medium | High | ✅ Complete |
| **3.2** | Attenuation | Medium | High | ✅ Complete |
| **4** | Inverse Problems | High | Very High | 🔲 Not started |
| **5** | Performance | Medium | High | 🔲 Not started |
| **6.1** | Finance/Darcy/NS | Low-Medium | Medium | 🔲 Not started |
| **6.2** | Maxwell/GR | High | Medium | 🔲 Not started |
| **7** | Theory Appendix | Low | Medium | 🔲 Not started |

---

## Classes to AVOID

These convenience classes hide Devito internals and must not appear in book code:

| Class | Why Avoid |
|-------|-----------|
| `SeismicModel`, `Model` | Hides Grid/Function setup |
| `AcousticWaveSolver` | Hides Operator construction |
| `AcquisitionGeometry` | Abstracts source/receiver setup |
| `PointSource`, `Receiver` | Wraps SparseTimeFunction |
| `RickerSource`, `GaborSource` | Hides wavelet generation |
| `demo_model()` | Model generation helper |

---

## Verification Requirements

Each solver must include:

1. **Exact polynomial solution** - FD scheme reproduces exactly
2. **MMS convergence test** - Verify expected convergence rate
3. **Conservation test** - Energy, mass, momentum where applicable
4. **Boundary condition test** - Values at boundaries
5. **Reference comparison** - scipy, analytical, or literature

---

## Completed Work Log

### 2026-01-29: Phase 1 Complete
- Created Chapter 6: Elliptic PDEs (984 lines)
- Added Burgers section to Chapter 5
- Created Chapter 7: Systems (SWE)
- 62 tests passing
- Commit: `ab9b38c0`

### 2026-01-29: Phase 2 Complete
- Created Chapter 8: High-Order Methods
- Added Section 7.3: Elastic Waves to Chapter 7
- 73 new tests (135 total)
- Added references for Fornberg, Tam-Webb
- Commit: `b9e017b3`

### 2026-01-29: Phase 3 Complete
- Added Sections 8.4-8.5: ADER and Staggered Grids to Chapter 8
- Added Sections 7.4-7.5: Viscoacoustic and Viscoelastic Waves to Chapter 7
- Created ADER solver (`ader_devito.py`) with Taylor expansion in time
- Created Staggered grid solver (`staggered_devito.py`) with `VectorTimeFunction`
- Created Viscoacoustic solvers with three rheological models: SLS, Kelvin-Voigt, Maxwell
- Created 3D Viscoelastic solver with `TensorTimeFunction` for stress/memory tensors
- 91 new tests (411 total)
- Fixed damping field creation for small grids
