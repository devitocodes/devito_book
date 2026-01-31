"""Distributed computing utilities for Devito workflows.

This module provides utilities for parallel execution of Devito
computations using Dask distributed. It is designed for embarrassingly
parallel workloads like shot-parallel seismic imaging.

Key features:
- Shot-parallel forward modeling
- Shot-parallel FWI gradient computation
- Integration with scipy.optimize
- Pickling utilities for Devito objects

Usage:
    from src.distributed import (
        create_local_cluster,
        forward_shot,
        parallel_forward_modeling,
        parallel_fwi_gradient,
        fwi_gradient_single_shot,
    )

    # Create cluster
    cluster, client = create_local_cluster(n_workers=4)

    # Parallel forward modeling
    shots = parallel_forward_modeling(
        client=client,
        velocity=vp,
        src_positions=sources,
        rec_coords=receivers,
        nt=2001,
        dt=0.5,
        f0=0.010,
        extent=(1000., 1000.),
    )

    # Clean up
    client.close()
    cluster.close()

Note:
    All functions that are submitted to Dask workers create Devito
    objects inside the function to avoid serialization issues with
    compiled operators.
"""

from .dask_utils import (
    FGPair,
    create_local_cluster,
    forward_shot,
    fwi_gradient_single_shot,
    parallel_forward_modeling,
    parallel_fwi_gradient,
    ricker_wavelet,
    sum_fg_pairs,
)

__all__ = [
    "FGPair",
    "create_local_cluster",
    "forward_shot",
    "fwi_gradient_single_shot",
    "parallel_forward_modeling",
    "parallel_fwi_gradient",
    "ricker_wavelet",
    "sum_fg_pairs",
]
