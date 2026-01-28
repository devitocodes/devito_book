"""Common utilities shared across chapters."""

from ..plotting import (
    set_seed,
    RANDOM_SEED,
    get_color_scheme,
    create_solution_plot,
    create_convergence_plot,
    create_animation_frames,
    save_figure,
    create_2d_heatmap,
)

from ..verification import (
    verify_identity,
    check_stencil_order,
    verify_pde_solution,
    numerical_verify,
    convergence_test,
    verify_stability_condition,
)

from ..display import (
    show_eq,
    show_eq_aligned,
    show_derivation,
    inline_latex,
)

__all__ = [
    'RANDOM_SEED',
    'check_stencil_order',
    'convergence_test',
    'create_2d_heatmap',
    'create_animation_frames',
    'create_convergence_plot',
    'create_solution_plot',
    'get_color_scheme',
    'inline_latex',
    'numerical_verify',
    'save_figure',
    'set_seed',
    'show_derivation',
    'show_eq',
    'show_eq_aligned',
    'verify_identity',
    'verify_pde_solution',
    'verify_stability_condition',
]
