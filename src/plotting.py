"""Plotting utilities for reproducible visualizations.

All plots use a fixed random seed and consistent styling for reproducibility.
Supports both Matplotlib (for static/PDF) and Plotly (for interactive HTML).

Usage:
    from src.plotting import create_solution_plot, create_convergence_plot

    # Ensure reproducibility
    set_seed()

    # Create solution plot
    fig = create_solution_plot(x, u_numerical, u_exact, title="Heat Equation")
"""

import warnings

import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility.

    Call this at the start of any notebook or script that uses randomness.

    Parameters
    ----------
    seed : int
        Random seed value (default: 42)
    """
    np.random.seed(seed)

    # Also try to set Python's random module if imported
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


# =============================================================================
# Color Schemes
# =============================================================================

# Professional color palette (colorblind-friendly)
COLORS = {
    'numerical': '#1f77b4',    # Blue
    'exact': '#ff7f0e',        # Orange
    'error': '#d62728',        # Red
    'initial': '#2ca02c',      # Green
    'boundary': '#9467bd',     # Purple
    'grid': '#7f7f7f',         # Gray
    'highlight': '#e377c2',    # Pink
}

# Alternative colorblind-safe palette (IBM Design)
COLORS_ACCESSIBLE = {
    'numerical': '#648FFF',    # Blue
    'exact': '#FFB000',        # Amber
    'error': '#DC267F',        # Magenta
    'initial': '#785EF0',      # Purple
    'boundary': '#FE6100',     # Orange
}


def get_color_scheme(name: str = 'default') -> dict:
    """Get a named color scheme.

    Parameters
    ----------
    name : str
        'default' or 'accessible'

    Returns
    -------
    dict
        Color mapping dictionary
    """
    if name == 'accessible':
        return COLORS_ACCESSIBLE
    return COLORS


# =============================================================================
# Matplotlib Plotting (for PDF output)
# =============================================================================

def create_solution_plot(
    x: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: np.ndarray | None = None,
    title: str = "Solution",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    figsize: tuple[int, int] = (8, 5),
    show_error: bool = False,
    backend: str = 'matplotlib',
):
    """Create a solution comparison plot.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    u_numerical : np.ndarray
        Numerical solution
    u_exact : np.ndarray, optional
        Exact/analytical solution for comparison
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size (width, height) in inches
    show_error : bool
        If True and u_exact provided, show error subplot
    backend : str
        'matplotlib' or 'plotly'

    Returns
    -------
    figure object
        Matplotlib Figure or Plotly Figure
    """
    if backend == 'plotly':
        return _create_solution_plot_plotly(
            x, u_numerical, u_exact, title, xlabel, ylabel, show_error
        )

    import matplotlib.pyplot as plt

    colors = get_color_scheme()

    if show_error and u_exact is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5),
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Main solution plot
    ax1.plot(x, u_numerical, '-', color=colors['numerical'],
             linewidth=2, label='Numerical')

    if u_exact is not None:
        ax1.plot(x, u_exact, '--', color=colors['exact'],
                 linewidth=2, label='Exact')

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Error subplot
    if ax2 is not None and u_exact is not None:
        error = u_numerical - u_exact
        ax2.plot(x, error, '-', color=colors['error'], linewidth=1.5)
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def _create_solution_plot_plotly(
    x: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: np.ndarray | None,
    title: str,
    xlabel: str,
    ylabel: str,
    show_error: bool,
):
    """Create solution plot using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        warnings.warn("Plotly not available, falling back to matplotlib", stacklevel=2)
        return create_solution_plot(
            x, u_numerical, u_exact, title, xlabel, ylabel,
            show_error=show_error, backend='matplotlib'
        )

    colors = get_color_scheme()

    if show_error and u_exact is not None:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                           vertical_spacing=0.1)
    else:
        fig = go.Figure()

    # Numerical solution
    fig.add_trace(go.Scatter(
        x=x, y=u_numerical,
        mode='lines',
        name='Numerical',
        line=dict(color=colors['numerical'], width=2),
    ), row=1 if show_error and u_exact is not None else None,
       col=1 if show_error and u_exact is not None else None)

    # Exact solution
    if u_exact is not None:
        fig.add_trace(go.Scatter(
            x=x, y=u_exact,
            mode='lines',
            name='Exact',
            line=dict(color=colors['exact'], width=2, dash='dash'),
        ), row=1 if show_error else None, col=1 if show_error else None)

    # Error subplot
    if show_error and u_exact is not None:
        error = u_numerical - u_exact
        fig.add_trace(go.Scatter(
            x=x, y=error,
            mode='lines',
            name='Error',
            line=dict(color=colors['error'], width=1.5),
            showlegend=False,
        ), row=2, col=1)

        fig.update_yaxes(title_text='Error', row=2, col=1)
        fig.update_xaxes(title_text=xlabel, row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        hovermode='x unified',
    )

    return fig


def create_convergence_plot(
    grid_sizes: np.ndarray,
    errors: np.ndarray,
    expected_order: float,
    title: str = "Convergence Study",
    xlabel: str = r"Grid size $N$",
    ylabel: str = "Error",
    figsize: tuple[int, int] = (8, 6),
    backend: str = 'matplotlib',
):
    """Create a log-log convergence plot.

    Parameters
    ----------
    grid_sizes : np.ndarray
        Array of grid sizes (N values)
    errors : np.ndarray
        Corresponding error values
    expected_order : float
        Expected convergence order (for reference line)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    backend : str
        'matplotlib' or 'plotly'

    Returns
    -------
    figure object
    """
    if backend == 'plotly':
        return _create_convergence_plot_plotly(
            grid_sizes, errors, expected_order, title, xlabel, ylabel
        )

    import matplotlib.pyplot as plt

    colors = get_color_scheme()

    fig, ax = plt.subplots(figsize=figsize)

    # Measured errors
    ax.loglog(grid_sizes, errors, 'o-', color=colors['numerical'],
              linewidth=2, markersize=8, label='Computed error')

    # Reference line
    h = 1.0 / grid_sizes
    ref_error = errors[0] * (h / h[0])**expected_order
    ax.loglog(grid_sizes, ref_error, '--', color=colors['grid'],
              linewidth=1.5, label=f'O(h^{expected_order:.1f})')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Compute and display observed order
    log_h = np.log(1.0 / grid_sizes)
    log_err = np.log(errors)
    observed_order = np.polyfit(log_h, log_err, 1)[0]
    ax.text(0.05, 0.05, f'Observed order: {observed_order:.2f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def _create_convergence_plot_plotly(
    grid_sizes: np.ndarray,
    errors: np.ndarray,
    expected_order: float,
    title: str,
    xlabel: str,
    ylabel: str,
):
    """Create convergence plot using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn("Plotly not available, falling back to matplotlib", stacklevel=2)
        return create_convergence_plot(
            grid_sizes, errors, expected_order, title, xlabel, ylabel,
            backend='matplotlib'
        )

    colors = get_color_scheme()

    fig = go.Figure()

    # Measured errors
    fig.add_trace(go.Scatter(
        x=grid_sizes, y=errors,
        mode='lines+markers',
        name='Computed error',
        line=dict(color=colors['numerical'], width=2),
        marker=dict(size=10),
    ))

    # Reference line
    h = 1.0 / grid_sizes
    ref_error = errors[0] * (h / h[0])**expected_order
    fig.add_trace(go.Scatter(
        x=grid_sizes, y=ref_error,
        mode='lines',
        name=f'O(h^{expected_order:.1f})',
        line=dict(color=colors['grid'], width=2, dash='dash'),
    ))

    # Compute observed order
    log_h = np.log(1.0 / np.array(grid_sizes))
    log_err = np.log(np.array(errors))
    observed_order = np.polyfit(log_h, log_err, 1)[0]

    fig.update_layout(
        title=f'{title} (Observed order: {observed_order:.2f})',
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
    )

    return fig


def create_animation_frames(
    x: np.ndarray,
    u_history: np.ndarray,
    times: np.ndarray,
    skip: int = 1,
    title_template: str = "Solution at t = {t:.3f}",
    backend: str = 'matplotlib',
) -> list:
    """Create animation frames for time-dependent solutions.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid
    u_history : np.ndarray
        Solution history, shape (n_times, n_x)
    times : np.ndarray
        Time values
    skip : int
        Frame skip (use every skip-th frame)
    title_template : str
        Title format string with {t} placeholder
    backend : str
        'matplotlib' or 'plotly'

    Returns
    -------
    list
        List of figure frames
    """
    frames = []
    indices = range(0, len(times), skip)

    if backend == 'plotly':
        try:
            import plotly.graph_objects as go

            # Create animated figure
            fig = go.Figure(
                data=[go.Scatter(x=x, y=u_history[0], mode='lines',
                                line=dict(color=COLORS['numerical'], width=2))],
                layout=go.Layout(
                    xaxis=dict(range=[x.min(), x.max()]),
                    yaxis=dict(range=[u_history.min(), u_history.max()]),
                    title=title_template.format(t=times[0]),
                    updatemenus=[dict(
                        type='buttons',
                        buttons=[dict(label='Play',
                                     method='animate',
                                     args=[None, {'frame': {'duration': 50}}])]
                    )]
                ),
                frames=[go.Frame(
                    data=[go.Scatter(x=x, y=u_history[i])],
                    name=str(i),
                    layout=go.Layout(title_text=title_template.format(t=times[i]))
                ) for i in indices]
            )
            return fig

        except ImportError:
            backend = 'matplotlib'

    # Matplotlib frames
    import matplotlib.pyplot as plt

    for i in indices:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, u_history[i], '-', color=COLORS['numerical'], linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(title_template.format(t=times[i]))
        ax.set_ylim(u_history.min(), u_history.max())
        ax.grid(True, alpha=0.3)
        frames.append(fig)
        plt.close(fig)

    return frames


def save_figure(
    fig,
    filename: str,
    dpi: int = 150,
    bbox_inches: str = 'tight',
):
    """Save figure to file with consistent settings.

    Parameters
    ----------
    fig : matplotlib Figure or plotly Figure
        Figure to save
    filename : str
        Output filename (with extension)
    dpi : int
        Resolution for raster formats
    bbox_inches : str
        Bounding box setting for matplotlib
    """
    if hasattr(fig, 'savefig'):
        # Matplotlib
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    elif hasattr(fig, 'write_image'):
        # Plotly
        fig.write_image(filename, scale=2)
    elif hasattr(fig, 'write_html'):
        # Plotly (HTML)
        fig.write_html(filename)
    else:
        raise TypeError(f"Unknown figure type: {type(fig)}")


def create_2d_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    title: str = "2D Solution",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "u(x, y)",
    figsize: tuple[int, int] = (8, 6),
    backend: str = 'matplotlib',
):
    """Create a 2D heatmap visualization.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays
    u : np.ndarray
        2D solution array, shape (len(y), len(x))
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    colorbar_label : str
        Colorbar label
    figsize : tuple
        Figure size
    backend : str
        'matplotlib' or 'plotly'

    Returns
    -------
    figure object
    """
    if backend == 'plotly':
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                x=x, y=y, z=u,
                colorscale='Viridis',
                colorbar=dict(title=colorbar_label),
            ))
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template='plotly_white',
            )
            return fig

        except ImportError:
            backend = 'matplotlib'

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(x, y, u, shading='auto', cmap='viridis')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'COLORS',
    'COLORS_ACCESSIBLE',
    'RANDOM_SEED',
    'create_2d_heatmap',
    'create_animation_frames',
    'create_convergence_plot',
    'create_solution_plot',
    'get_color_scheme',
    'save_figure',
    'set_seed',
]
