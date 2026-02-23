#!/usr/bin/env python
"""Generate comparison figures for the ABC chapter.

This script runs all ABC methods on the same 2D test problem and produces
figures for the book. It requires Devito to be installed.

Usage:
    python src/wave/generate_abc_figures.py

Output figures are saved to chapters/wave/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

FIGURE_DIR = project_root / "chapters" / "wave" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def gaussian_ic(X, Y, x0=1.0, y0=1.0, sigma=0.1):
    """Gaussian point-source initial condition."""
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))


def run_all_methods():
    """Run all ABC methods and return results dict."""
    from src.wave.abc_methods import solve_wave_2d_abc

    Lx, Ly = 2.0, 2.0
    Nx, Ny = 100, 100
    T = 1.5
    CFL = 0.5
    pad = 20

    methods = ['dirichlet', 'first_order', 'damping', 'pml', 'higdon', 'habc']
    results = {}

    for method in methods:
        print(f"Running {method}...")
        kw = dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, T=T, CFL=CFL,
                  I=gaussian_ic, abc_type=method, pad_width=pad,
                  save_history=True)
        # HABC uses a thinner layer
        if method == 'habc':
            kw['pad_width'] = 10
        results[method] = solve_wave_2d_abc(**kw)

    return results


def fig_reflection_problem(results):
    """Figure showing Dirichlet BC reflection artifacts."""
    r = results['dirichlet']

    # Find a time step with strong reflections
    nt = r.u_history.shape[0]
    idx = min(nt - 1, int(0.8 * nt))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(
        r.u_history[idx].T, origin='lower',
        extent=[0, 2, 0, 2], cmap='RdBu',
        vmin=-0.5, vmax=0.5,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Dirichlet BC: strong reflections (t = {r.t_history[idx]:.2f})')
    plt.colorbar(im, ax=ax, label='u')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_abc_reflection_problem.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_abc_reflection_problem.png")


def fig_damping_snapshots(results):
    """Time snapshots with damping layer."""
    r = results['damping']
    nt = r.u_history.shape[0]
    indices = [int(f * nt) for f in [0.1, 0.3, 0.5, 0.8]]
    indices = [min(i, nt - 1) for i in indices]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, idx in zip(axes, indices):
        im = ax.imshow(
            r.u_history[idx].T, origin='lower',
            extent=[0, 2, 0, 2], cmap='RdBu',
            vmin=-0.5, vmax=0.5,
        )
        ax.set_title(f't = {r.t_history[idx]:.2f}')
        ax.set_xlabel('x')
        if ax == axes[0]:
            ax.set_ylabel('y')
    fig.suptitle('Damping Layer ABC', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_abc_damping_snapshots.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_abc_damping_snapshots.png")


def fig_pml_snapshots(results):
    """Time snapshots with PML."""
    r = results['pml']
    nt = r.u_history.shape[0]
    indices = [int(f * nt) for f in [0.1, 0.3, 0.5, 0.8]]
    indices = [min(i, nt - 1) for i in indices]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, idx in zip(axes, indices):
        im = ax.imshow(
            r.u_history[idx].T, origin='lower',
            extent=[0, 2, 0, 2], cmap='RdBu',
            vmin=-0.5, vmax=0.5,
        )
        ax.set_title(f't = {r.t_history[idx]:.2f}')
        ax.set_xlabel('x')
        if ax == axes[0]:
            ax.set_ylabel('y')
    fig.suptitle('PML ABC', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_abc_pml_snapshots.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_abc_pml_snapshots.png")


def fig_comparison(results):
    """Reflection energy vs. method comparison."""
    from src.wave.abc_methods import measure_reflection

    methods = ['dirichlet', 'first_order', 'damping', 'pml', 'higdon', 'habc']
    labels = ['Dirichlet', 'First-order', 'Damping', 'PML', 'Higdon P=2', 'HABC']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    reflections = []

    for method in methods:
        R = measure_reflection(results[method])
        reflections.append(R)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bars = ax.bar(labels, reflections, color=colors)
    ax.set_ylabel('Reflection coefficient')
    ax.set_title('ABC Method Comparison (2D point source)')
    ax.set_ylim(0, max(reflections) * 1.2 if max(reflections) > 0 else 1.0)

    for bar, val in zip(bars, reflections):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_abc_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_abc_comparison.png")


def fig_parameter_study():
    """Damping layer: effect of width and polynomial order."""
    from src.wave.abc_methods import measure_reflection, solve_wave_2d_abc

    widths = [5, 10, 15, 20, 30]
    orders = [1, 2, 3]
    Lx, Ly, Nx, Ny, T, CFL = 2.0, 2.0, 80, 80, 1.5, 0.5

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for order in orders:
        Rs = []
        for w in widths:
            result = solve_wave_2d_abc(
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, T=T, CFL=CFL,
                I=gaussian_ic, abc_type='damping',
                pad_width=w, damping_order=order,
            )
            R = measure_reflection(result)
            Rs.append(R)
        ax.plot(widths, Rs, 'o-', label=f'order p={order}')

    ax.set_xlabel('Layer width (grid cells)')
    ax.set_ylabel('Reflection coefficient')
    ax.set_title('Damping Layer: Effect of Width and Polynomial Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_abc_parameter_study.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_abc_parameter_study.png")


def main():
    print("Generating ABC comparison figures...")
    print(f"Output directory: {FIGURE_DIR}")

    results = run_all_methods()

    fig_reflection_problem(results)
    fig_damping_snapshots(results)
    fig_pml_snapshots(results)
    fig_comparison(results)
    fig_parameter_study()

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
