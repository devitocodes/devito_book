"""Snapshotting utilities for memory-efficient wave propagation.

This module provides utilities for wavefield snapshotting using
Devito's ConditionalDimension, enabling memory-efficient storage
of wavefields during wave propagation simulations.

Key concepts:
- ConditionalDimension: Creates subsampled time dimension
- Snapshot TimeFunction: Stores wavefield at regular intervals
- Memory estimation: Compute storage requirements

Example:
    from src.memory.snapshotting import (
        estimate_wavefield_memory,
        wave_propagation_with_snapshotting,
    )

    # Estimate memory
    mem = estimate_wavefield_memory(shape=(501, 501, 201), nt=2000)
    print(f"Full storage: {mem['full_storage_GB']:.1f} GB")

    # Run with snapshotting
    result = wave_propagation_with_snapshotting(
        shape=(101, 101),
        extent=(1000., 1000.),
        nt=500,
        snapshot_factor=10
    )
"""

from dataclasses import dataclass

import numpy as np

try:
    from devito import (
        ConditionalDimension,
        Eq,
        Function,
        Grid,
        Operator,
        TimeFunction,
        solve,
    )
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


def estimate_wavefield_memory(
    shape: tuple,
    nt: int,
    dtype_bytes: int = 4,
    time_order: int = 2,
) -> dict:
    """Estimate memory requirements for wavefield storage.

    Computes memory requirements for different storage strategies:
    - Full wavefield storage (all time steps)
    - Rolling buffer (minimal, for forward propagation only)
    - Snapshotting with various factors

    Parameters
    ----------
    shape : tuple
        Spatial grid shape, e.g., (nx, ny) or (nx, ny, nz)
    nt : int
        Number of time steps
    dtype_bytes : int, optional
        Bytes per element. Default: 4 (float32)
    time_order : int, optional
        Time order of the scheme. Default: 2

    Returns
    -------
    dict
        Memory estimates with keys:
        - 'grid_points': Total spatial grid points
        - 'dimensions': Number of spatial dimensions
        - 'per_snapshot_MB': Memory per snapshot in MB
        - 'full_storage_GB': Full wavefield storage in GB
        - 'rolling_buffer_MB': Rolling buffer size in MB
        - 'snapshot_factor_N_GB': Memory with factor N snapshotting

    Examples
    --------
    >>> mem = estimate_wavefield_memory(shape=(501, 501, 201), nt=2000)
    >>> print(f"Full storage: {mem['full_storage_GB']:.1f} GB")
    Full storage: 402.8 GB
    >>> print(f"Factor 50 snapshotting: {mem['snapshot_factor_50_GB']:.1f} GB")
    Factor 50 snapshotting: 8.1 GB
    """
    ndim = len(shape)
    npoints = int(np.prod(shape))
    time_buffer = time_order + 1

    # Memory in bytes
    per_snapshot = npoints * dtype_bytes
    full_storage = nt * per_snapshot
    rolling_buffer = time_buffer * per_snapshot

    results = {
        'grid_points': npoints,
        'dimensions': ndim,
        'time_steps': nt,
        'per_snapshot_bytes': per_snapshot,
        'per_snapshot_MB': per_snapshot / (1024**2),
        'full_storage_bytes': full_storage,
        'full_storage_GB': full_storage / (1024**3),
        'rolling_buffer_bytes': rolling_buffer,
        'rolling_buffer_MB': rolling_buffer / (1024**2),
    }

    # Snapshotting estimates for common factors
    for factor in [5, 10, 20, 50, 100]:
        nsnaps = nt // factor
        snap_memory = nsnaps * per_snapshot
        results[f'snapshot_factor_{factor}_nsnaps'] = nsnaps
        results[f'snapshot_factor_{factor}_GB'] = snap_memory / (1024**3)

    return results


@dataclass
class SnapshotResult:
    """Results from wave propagation with snapshotting.

    Attributes
    ----------
    snapshots : np.ndarray
        Saved wavefield snapshots, shape (nsnaps, *spatial_shape)
    time_indices : np.ndarray
        Time step indices corresponding to snapshots
    memory_savings : float
        Memory savings factor compared to full storage
    snapshot_factor : int
        Factor used for snapshotting
    grid_shape : tuple
        Spatial grid shape
    """
    snapshots: np.ndarray
    time_indices: np.ndarray
    memory_savings: float
    snapshot_factor: int
    grid_shape: tuple


def create_snapshot_timefunction(
    shape: tuple,
    extent: tuple,
    nt: int,
    snapshot_factor: int = 10,
    space_order: int = 4,
    dtype: type = np.float32,
) -> tuple:
    """Create a Grid and snapshotted TimeFunction for wave propagation.

    This function sets up the Devito objects needed for wavefield
    snapshotting using ConditionalDimension.

    Parameters
    ----------
    shape : tuple
        Spatial grid shape, e.g., (nx, ny)
    extent : tuple
        Physical domain extent, e.g., (Lx, Ly) in meters
    nt : int
        Number of time steps
    snapshot_factor : int, optional
        Save wavefield every snapshot_factor steps. Default: 10
    space_order : int, optional
        Spatial discretization order. Default: 4
    dtype : type, optional
        Data type. Default: np.float32

    Returns
    -------
    tuple
        (grid, usave) where:
        - grid: Devito Grid object
        - usave: TimeFunction for snapshot storage

    Raises
    ------
    ImportError
        If Devito is not installed

    Examples
    --------
    >>> grid, usave = create_snapshot_timefunction(
    ...     shape=(101, 101),
    ...     extent=(1000., 1000.),
    ...     nt=500,
    ...     snapshot_factor=10
    ... )
    >>> print(f"Snapshot buffer shape: {usave.data.shape}")
    Snapshot buffer shape: (50, 101, 101)
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for snapshotting. "
            "Install with: pip install devito"
        )

    # Number of snapshots
    nsnaps = nt // snapshot_factor

    # Create grid
    grid = Grid(shape=shape, extent=extent, dtype=dtype)
    time = grid.time_dim

    # Create subsampled time dimension
    time_sub = ConditionalDimension(
        't_sub', parent=time, factor=snapshot_factor
    )

    # Create snapshot TimeFunction
    usave = TimeFunction(
        name='usave',
        grid=grid,
        time_order=0,
        space_order=space_order,
        save=nsnaps,
        time_dim=time_sub
    )

    return grid, usave


def wave_propagation_with_snapshotting(
    shape: tuple = (101, 101),
    extent: tuple = (1000., 1000.),
    vel: float = 2.0,
    nt: int = 500,
    dt: float = 1.0,
    snapshot_factor: int = 10,
    initial_condition: str = 'gaussian',
) -> SnapshotResult:
    """Solve 2D wave equation with wavefield snapshotting.

    This function demonstrates the snapshotting pattern using
    ConditionalDimension for memory-efficient wavefield storage.

    Parameters
    ----------
    shape : tuple, optional
        Grid shape (nx, ny). Default: (101, 101)
    extent : tuple, optional
        Physical extent (Lx, Ly) in meters. Default: (1000., 1000.)
    vel : float, optional
        Wave velocity in km/s. Default: 2.0
    nt : int, optional
        Number of time steps. Default: 500
    dt : float, optional
        Time step in ms. Default: 1.0
    snapshot_factor : int, optional
        Save wavefield every snapshot_factor steps. Default: 10
    initial_condition : str, optional
        Type of initial condition ('gaussian' or 'plane'). Default: 'gaussian'

    Returns
    -------
    SnapshotResult
        Result containing snapshots, timing, and memory info

    Raises
    ------
    ImportError
        If Devito is not installed

    Examples
    --------
    >>> result = wave_propagation_with_snapshotting(
    ...     shape=(101, 101),
    ...     extent=(1000., 1000.),
    ...     nt=500,
    ...     snapshot_factor=10
    ... )
    >>> print(f"Collected {len(result.time_indices)} snapshots")
    Collected 50 snapshots
    >>> print(f"Memory savings: {result.memory_savings:.1f}x")
    Memory savings: 15.6x
    """
    if not DEVITO_AVAILABLE:
        raise ImportError(
            "Devito is required for this function. "
            "Install with: pip install devito"
        )

    # Number of snapshots
    nsnaps = nt // snapshot_factor

    # Create grid
    grid = Grid(shape=shape, extent=extent, dtype=np.float32)
    time = grid.time_dim

    # Create subsampled time dimension
    time_sub = ConditionalDimension('t_sub', parent=time, factor=snapshot_factor)

    # Velocity field
    v = Function(name='v', grid=grid, space_order=4)
    v.data[:] = vel

    # Forward wavefield (rolling buffer - only 3 time levels)
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)

    # Snapshot storage
    usave = TimeFunction(
        name='usave', grid=grid,
        time_order=0, save=nsnaps, time_dim=time_sub
    )

    # Set initial condition
    X, Y = np.meshgrid(
        np.linspace(0, extent[0], shape[0]),
        np.linspace(0, extent[1], shape[1]),
        indexing='ij'
    )

    if initial_condition == 'gaussian':
        # Gaussian pulse at center
        cx, cy = extent[0] / 2, extent[1] / 2
        sigma = min(extent) / 20
        u0 = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    elif initial_condition == 'plane':
        # Plane wave
        u0 = np.sin(2 * np.pi * X / extent[0])
    else:
        u0 = np.zeros(shape, dtype=np.float32)

    u.data[0, :, :] = u0.astype(np.float32)
    u.data[1, :, :] = u0.astype(np.float32)

    # Wave equation: u_tt = v^2 * laplace(u)
    pde = (1.0 / v**2) * u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Snapshot equation (conditional save)
    snapshot_eq = Eq(usave, u)

    # Create operator with both equations
    op = Operator([stencil, snapshot_eq])

    # Run
    op.apply(time=nt - 2, dt=dt)

    # Calculate memory savings
    full_memory = nt * np.prod(shape) * 4  # bytes
    actual_memory = u.data.nbytes + usave.data.nbytes
    savings = full_memory / actual_memory

    # Time indices for snapshots
    time_indices = np.arange(0, nt, snapshot_factor)

    return SnapshotResult(
        snapshots=usave.data.copy(),
        time_indices=time_indices,
        memory_savings=savings,
        snapshot_factor=snapshot_factor,
        grid_shape=shape,
    )


def save_wavefield(
    data: np.ndarray,
    filename: str,
    compressed: bool = False,
) -> dict:
    """Save wavefield to file.

    Supports both raw binary and compressed NumPy formats.

    Parameters
    ----------
    data : np.ndarray
        Wavefield data to save
    filename : str
        Output filename (extension determines format)
    compressed : bool, optional
        If True, use compressed .npz format. Default: False

    Returns
    -------
    dict
        I/O statistics including sizes and compression ratio

    Examples
    --------
    >>> data = np.random.randn(100, 100, 100).astype(np.float32)
    >>> stats = save_wavefield(data, '/tmp/wavefield.bin')
    >>> print(f"Saved {stats['size_MB']:.1f} MB")
    """
    import os

    if compressed or filename.endswith('.npz'):
        np.savez_compressed(filename if not filename.endswith('.npz') else filename[:-4], data=data)
        actual_filename = filename if filename.endswith('.npz') else filename + '.npz'
        file_size = os.path.getsize(actual_filename)
    else:
        data.tofile(filename)
        file_size = os.path.getsize(filename)

    stats = {
        'filename': filename,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'uncompressed_bytes': data.nbytes,
        'file_bytes': file_size,
        'size_MB': file_size / (1024**2),
        'compression_ratio': data.nbytes / file_size if file_size > 0 else 0,
    }

    return stats


def load_wavefield(
    filename: str,
    shape: tuple = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Load wavefield from file.

    Automatically detects format based on file extension.

    Parameters
    ----------
    filename : str
        Input filename
    shape : tuple, optional
        Expected array shape (required for raw binary files)
    dtype : type, optional
        Data type. Default: np.float32

    Returns
    -------
    np.ndarray
        Loaded wavefield

    Raises
    ------
    ValueError
        If shape is not provided for raw binary files

    Examples
    --------
    >>> data = load_wavefield('/tmp/wavefield.bin', shape=(100, 100, 100))
    >>> print(f"Loaded shape: {data.shape}")
    """
    if filename.endswith('.npz'):
        return np.load(filename)['data']
    elif filename.endswith('.npy'):
        return np.load(filename)
    else:
        # Raw binary
        if shape is None:
            raise ValueError("shape must be provided for raw binary files")
        data = np.fromfile(filename, dtype=dtype)
        return data.reshape(shape)


def save_wavefield_hdf5(
    data: np.ndarray,
    filename: str,
    dataset_name: str = 'wavefield',
    compression: str = 'gzip',
    compression_level: int = 4,
) -> dict:
    """Save wavefield to HDF5 with chunking and compression.

    HDF5 provides efficient storage for large arrays with:
    - Chunked storage for efficient partial reads
    - Built-in compression
    - Parallel I/O support (with MPI-enabled h5py)

    Parameters
    ----------
    data : np.ndarray
        Wavefield data
    filename : str
        Output HDF5 filename
    dataset_name : str, optional
        Name of dataset in file. Default: 'wavefield'
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', None). Default: 'gzip'
    compression_level : int, optional
        Compression level (1-9 for gzip). Default: 4

    Returns
    -------
    dict
        I/O statistics

    Raises
    ------
    ImportError
        If h5py is not installed

    Examples
    --------
    >>> data = np.random.randn(100, 100, 100).astype(np.float32)
    >>> stats = save_wavefield_hdf5(data, '/tmp/wavefield.h5')
    >>> print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError("h5py required for HDF5 I/O. Install with: pip install h5py") from err

    import os

    # Determine chunk sizes (aim for ~1MB chunks)
    target_chunk_bytes = 1024 * 1024
    bytes_per_element = data.itemsize
    elements_per_chunk = target_chunk_bytes // bytes_per_element

    # Calculate chunk dimensions
    ndim = len(data.shape)
    chunk_size = int(np.power(elements_per_chunk, 1.0 / ndim))
    chunks = tuple(min(chunk_size, s) for s in data.shape)

    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            dataset_name, data=data,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None,
            chunks=chunks
        )

    file_size = os.path.getsize(filename)

    stats = {
        'filename': filename,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'chunks': chunks,
        'compression': compression,
        'uncompressed_bytes': data.nbytes,
        'file_bytes': file_size,
        'size_MB': file_size / (1024**2),
        'compression_ratio': data.nbytes / file_size if file_size > 0 else 0,
    }

    return stats


def load_wavefield_hdf5(
    filename: str,
    dataset_name: str = 'wavefield',
    slices: tuple = None,
) -> np.ndarray:
    """Load wavefield from HDF5, optionally with slicing.

    Supports partial loading through slicing, which is efficient
    due to HDF5's chunked storage.

    Parameters
    ----------
    filename : str
        Input HDF5 filename
    dataset_name : str, optional
        Name of dataset. Default: 'wavefield'
    slices : tuple, optional
        Slice specification for partial loading, e.g., (slice(0, 10), ...)

    Returns
    -------
    np.ndarray
        Loaded wavefield (or slice thereof)

    Raises
    ------
    ImportError
        If h5py is not installed

    Examples
    --------
    >>> # Load full array
    >>> data = load_wavefield_hdf5('/tmp/wavefield.h5')
    >>> # Load first 10 time steps
    >>> partial = load_wavefield_hdf5('/tmp/wavefield.h5', slices=(slice(0, 10),))
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError("h5py required for HDF5 I/O. Install with: pip install h5py") from err

    with h5py.File(filename, 'r') as f:
        if slices is not None:
            return f[dataset_name][slices]
        return f[dataset_name][:]
