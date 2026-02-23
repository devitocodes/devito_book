#!/usr/bin/env python
"""
Generate stencil figures for the wave equation chapter.

This script generates three stencil diagrams:
- stencil_n_interior.png: Standard 5-point stencil at interior point
- stencil_n0_interior.png: Modified 4-point stencil for first time step
- stencil_n_left.png: Modified stencil at left boundary (Neumann condition)

Each figure includes a legend explaining:
- Filled blue circles: Known values (computed at previous time steps)
- Empty black circle: Unknown value (to be computed)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def create_stencil_figure(
    known_points,
    unknown_point,
    title,
    filename,
    xlim=(0, 5),
    ylim=(0, 5),
):
    """
    Create a stencil figure with legend.

    Parameters
    ----------
    known_points : list of tuples
        List of (i, n) coordinates for known values
    unknown_point : tuple
        (i, n) coordinate for the unknown value
    title : str
        Figure title
    filename : str
        Output filename
    xlim, ylim : tuples
        Axis limits
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot grid
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(range(xlim[0], xlim[1] + 1))
    ax.set_yticks(range(ylim[0], ylim[1] + 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    # Plot known points (filled blue circles)
    for i, n in known_points:
        circle = plt.Circle(
            (i, n), 0.15, fill=True, color='blue', linewidth=2
        )
        ax.add_patch(circle)

    # Plot unknown point (empty black circle)
    i, n = unknown_point
    circle = plt.Circle(
        (i, n), 0.15, fill=False, color='black', linewidth=2
    )
    ax.add_patch(circle)

    # Labels
    ax.set_xlabel('index i', fontsize=12)
    ax.set_ylabel('index n', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Create legend
    known_patch = mpatches.Patch(
        facecolor='blue', edgecolor='blue',
        label='Known (from previous time steps)'
    )
    unknown_patch = mpatches.Patch(
        facecolor='white', edgecolor='black',
        label='Unknown (to be computed)'
    )
    ax.legend(
        handles=[known_patch, unknown_patch],
        loc='upper right',
        fontsize=10,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {filename}")


def main():
    # Figure 1: Standard interior stencil (5 points)
    # Computing u[2]^3 from u[1]^2, u[2]^2, u[3]^2, u[2]^1
    create_stencil_figure(
        known_points=[(1, 2), (2, 2), (3, 2), (2, 1)],
        unknown_point=(2, 3),
        title='Stencil at interior point',
        filename='stencil_n_interior.png'
    )

    # Figure 2: First time step stencil (4 points, no n-1 level)
    # Computing u[2]^1 from u[1]^0, u[2]^0, u[3]^0
    create_stencil_figure(
        known_points=[(1, 0), (2, 0), (3, 0)],
        unknown_point=(2, 1),
        title='Stencil at interior point (first time step)',
        filename='stencil_n0_interior.png'
    )

    # Figure 3: Left boundary stencil (Neumann condition)
    # Computing u[0]^3 from u[0]^2, u[1]^2, u[0]^1
    create_stencil_figure(
        known_points=[(0, 2), (1, 2), (0, 1)],
        unknown_point=(0, 3),
        title='Stencil at boundary point (Neumann condition)',
        filename='stencil_n_left.png'
    )

    print("\nAll stencil figures generated successfully!")
    print("\nLegend explanation:")
    print("  - Filled blue circles: Known values (computed at previous time steps)")
    print("  - Empty black circle: Unknown value (to be computed)")


if __name__ == '__main__':
    main()
