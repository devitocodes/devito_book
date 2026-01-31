"""Memory management utilities for Devito wave simulations.

This module provides utilities for efficient wavefield storage,
snapshotting, and I/O operations in large-scale wave propagation
simulations.

Key features:
- Memory estimation for wavefield storage
- ConditionalDimension-based snapshotting
- I/O utilities for wavefield persistence
- Checkpointing support (via pyrevolve integration)

Example usage:
    from src.memory import (
        estimate_wavefield_memory,
        create_snapshot_timefunction,
        save_wavefield,
        load_wavefield,
    )

    # Estimate memory requirements
    mem = estimate_wavefield_memory(shape=(501, 501, 201), nt=2000)
    print(f"Full storage: {mem['full_storage_GB']:.1f} GB")

    # Create snapshotted TimeFunction
    grid, usave = create_snapshot_timefunction(
        shape=(101, 101),
        extent=(1000., 1000.),
        nt=500,
        snapshot_factor=10
    )
"""

from .snapshotting import (
    DEVITO_AVAILABLE,
    SnapshotResult,
    create_snapshot_timefunction,
    estimate_wavefield_memory,
    load_wavefield,
    load_wavefield_hdf5,
    save_wavefield,
    save_wavefield_hdf5,
    wave_propagation_with_snapshotting,
)

__all__ = [
    'DEVITO_AVAILABLE',
    'SnapshotResult',
    'create_snapshot_timefunction',
    'estimate_wavefield_memory',
    'load_wavefield',
    'load_wavefield_hdf5',
    'save_wavefield',
    'save_wavefield_hdf5',
    'wave_propagation_with_snapshotting',
]
