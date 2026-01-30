"""
On-the-fly discrete Fourier transform using Devito.

This module provides memory-efficient frequency-domain wavefield computation
by accumulating Fourier modes during time stepping, avoiding storage of
the full time history.

Theory
------
The DFT of a time series u(t_n) sampled at N time steps is:
    U(omega_k) = sum_{n=0}^{N-1} u(t_n) * exp(-i * omega_k * t_n) * dt

This can be computed incrementally:
    U_k^{n+1} = U_k^n + u(t_n) * exp(-i * omega_k * t_n) * dt

References
----------
Witte et al. (2019). "Compressive least-squares migration with on-the-fly
Fourier transforms", Geophysics, 84(5), R655-R672.
"""


import numpy as np

# Devito imports - will fail gracefully if not installed
try:
    from devito import (
        Dimension,
        Eq,
        Function,
        Grid,
        Inc,
        Operator,
        SparseTimeFunction,
        TimeFunction,
        solve,
    )
    from sympy import exp, pi
    DEVITO_AVAILABLE = True
except ImportError:
    DEVITO_AVAILABLE = False


def ricker_wavelet(
    t: np.ndarray,
    f0: float,
    t0: float = None
) -> np.ndarray:
    """
    Compute a Ricker wavelet (Mexican hat wavelet).

    Parameters
    ----------
    t : ndarray
        Time values
    f0 : float
        Peak (dominant) frequency in Hz
    t0 : float, optional
        Time shift. Default: 1.5/f0 (centers the wavelet)

    Returns
    -------
    wavelet : ndarray
        Ricker wavelet values

    Notes
    -----
    The Ricker wavelet is the negative second derivative of a Gaussian.
    It is commonly used as a seismic source wavelet.
    """
    if t0 is None:
        t0 = 1.5 / f0
    r = np.pi * f0 * (t - t0)
    return (1 - 2 * r**2) * np.exp(-r**2)


def run_otf_dft(
    nx: int = 101,
    ny: int = 101,
    nt: int = 500,
    freq: float = 10.0,
    f0: float = 15.0,
    velocity: float = 1500.0,
    extent: tuple[float, float] = (1000., 1000.)
) -> tuple[np.ndarray, dict]:
    """
    Run acoustic wave simulation with single-frequency on-the-fly DFT.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    nt : int
        Number of time steps
    freq : float
        Frequency (Hz) for DFT computation
    f0 : float
        Source peak frequency (Hz)
    velocity : float
        Acoustic velocity (m/s)
    extent : tuple
        Physical domain size (Lx, Ly) in meters

    Returns
    -------
    freq_mode : ndarray
        Complex Fourier mode, shape (nx, ny)
    info : dict
        Simulation parameters and metadata

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for on-the-fly DFT")

    # Grid setup
    grid = Grid(shape=(nx, ny), extent=extent)
    dx, dy = grid.spacing

    # Compute stable time step (conservative CFL for stability)
    dt = 0.5 * float(min(dx, dy)) / velocity

    # Wavefield
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)

    # Slowness squared model (m = 1/c^2) - standard seismic pattern
    m = Function(name='m', grid=grid)
    m.data[:] = 1.0 / velocity**2

    # Source setup
    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
    src.coordinates.data[:] = [[extent[0]/2, extent[1]/2]]
    time_values = np.arange(nt) * dt
    src.data[:, 0] = ricker_wavelet(time_values, f0)

    # Frequency mode storage (complex)
    freq_mode = Function(name='freq_mode', grid=grid, dtype=np.complex64)

    # Time dimension (use time_dim for DFT basis)
    time_dim = grid.time_dim
    dt_spacing = time_dim.spacing

    # Fourier basis: exp(-1j * omega * t * dt)
    omega = 2 * pi * freq
    basis = exp(-1j * omega * time_dim * dt_spacing)

    # PDE: m * u_tt - laplacian(u) = 0
    pde = m * u.dt2 - u.laplace
    update = Eq(u.forward, solve(pde, u.forward))

    # Source injection: src * dt^2 / m (standard seismic pattern)
    src_term = src.inject(field=u.forward, expr=src * dt_spacing**2 / m)

    # DFT accumulation
    dft_eq = Inc(freq_mode, basis * u)

    # Create and run operator with spacing map
    op = Operator([update, src_term, dft_eq], subs=grid.spacing_map)
    op(time_M=nt-1, dt=dt)

    info = {
        'nx': nx, 'ny': ny, 'nt': nt,
        'dx': float(dx), 'dy': float(dy), 'dt': dt,
        'freq': freq, 'f0': f0,
        'velocity': velocity,
        'extent': extent,
        'cfl': velocity * dt / min(dx, dy)
    }

    return freq_mode.data.copy(), info


def run_otf_dft_multifreq(
    nx: int = 101,
    ny: int = 101,
    nt: int = 500,
    frequencies: np.ndarray = None,
    f0: float = 15.0,
    velocity: float = 1500.0,
    extent: tuple[float, float] = (1000., 1000.)
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run acoustic wave simulation with multi-frequency on-the-fly DFT.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    nt : int
        Number of time steps
    frequencies : ndarray, optional
        Frequencies (Hz) for DFT. Default: [5, 10, 15, 20]
    f0 : float
        Source peak frequency (Hz)
    velocity : float
        Acoustic velocity (m/s)
    extent : tuple
        Physical domain size (Lx, Ly) in meters

    Returns
    -------
    freq_modes : ndarray
        Complex Fourier modes, shape (nfreq, nx, ny)
    frequencies : ndarray
        Frequency values
    info : dict
        Simulation parameters and metadata

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for on-the-fly DFT")

    if frequencies is None:
        frequencies = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
    else:
        frequencies = np.asarray(frequencies, dtype=np.float32)

    nfreq = len(frequencies)

    # Grid setup
    grid = Grid(shape=(nx, ny), extent=extent)
    dx, dy = grid.spacing

    # Compute stable time step (conservative CFL)
    dt = 0.5 * float(min(dx, dy)) / velocity

    # Wavefield
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)

    # Slowness squared model (m = 1/c^2)
    m = Function(name='m', grid=grid)
    m.data[:] = 1.0 / velocity**2

    # Source setup
    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
    src.coordinates.data[:] = [[extent[0]/2, extent[1]/2]]
    time_values = np.arange(nt) * dt
    src.data[:, 0] = ricker_wavelet(time_values, f0)

    # Frequency dimension
    f = Dimension(name='f')
    freqs = Function(name='freqs', dimensions=(f,), shape=(nfreq,),
                     dtype=np.float32)
    freqs.data[:] = frequencies

    # Multi-frequency mode storage
    freq_modes = Function(name='freq_modes', dtype=np.complex64,
                          dimensions=(f, *grid.dimensions),
                          shape=(nfreq, *grid.shape))

    # Time dimension (use time_dim for DFT basis)
    time_dim = grid.time_dim
    dt_spacing = time_dim.spacing

    # Vectorized Fourier basis
    omega = 2 * pi * freqs
    basis = exp(-1j * omega * time_dim * dt_spacing)

    # PDE: m * u_tt - laplacian(u) = 0
    pde = m * u.dt2 - u.laplace
    update = Eq(u.forward, solve(pde, u.forward))

    # Source injection: src * dt^2 / m
    src_term = src.inject(field=u.forward, expr=src * dt_spacing**2 / m)

    # DFT accumulation (broadcasts over frequency dimension)
    dft_eq = Inc(freq_modes, basis * u)

    # Create and run operator with spacing map
    op = Operator([update, src_term, dft_eq], subs=grid.spacing_map)
    op(time_M=nt-1, dt=dt)

    info = {
        'nx': nx, 'ny': ny, 'nt': nt,
        'dx': float(dx), 'dy': float(dy), 'dt': dt,
        'frequencies': frequencies.tolist(),
        'nfreq': nfreq,
        'f0': f0,
        'velocity': velocity,
        'extent': extent,
        'cfl': velocity * dt / min(dx, dy)
    }

    return freq_modes.data.copy(), frequencies, info


def compute_reference_dft(
    u_history: np.ndarray,
    frequencies: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute DFT from stored time history (reference implementation).

    This function computes the DFT directly from the full time history,
    serving as a reference for verifying the on-the-fly implementation.

    Parameters
    ----------
    u_history : ndarray
        Wavefield time history, shape (nt, nx, ny) or (nt, nx)
    frequencies : ndarray
        Frequencies (Hz) for DFT
    dt : float
        Time step

    Returns
    -------
    freq_modes : ndarray
        Complex Fourier modes, shape (nfreq, nx, ny) or (nfreq, nx)
    """
    nt = u_history.shape[0]
    spatial_shape = u_history.shape[1:]
    nfreq = len(frequencies)

    # Time array
    t = np.arange(nt) * dt

    # Initialize output
    freq_modes = np.zeros((nfreq,) + spatial_shape, dtype=np.complex64)

    # Compute DFT for each frequency
    for k, f in enumerate(frequencies):
        omega = 2 * np.pi * f
        # exp(-i * omega * t) integrated over time
        for n in range(nt):
            freq_modes[k] += u_history[n] * np.exp(-1j * omega * t[n]) * dt

    return freq_modes


def compare_otf_to_fft(
    nx: int = 51,
    ny: int = 51,
    nt: int = 200,
    frequencies: np.ndarray = None,
    rtol: float = 0.1
) -> tuple[bool, float, dict]:
    """
    Compare on-the-fly DFT to reference FFT-based computation.

    This function runs a simulation storing the full time history,
    then compares the on-the-fly DFT result to a post-hoc DFT
    computed from the stored history.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (keep small for memory)
    nt : int
        Number of time steps
    frequencies : ndarray, optional
        Frequencies to compare
    rtol : float
        Relative tolerance for comparison

    Returns
    -------
    passed : bool
        True if results match within tolerance
    max_error : float
        Maximum relative error
    details : dict
        Detailed comparison information

    Raises
    ------
    ImportError
        If Devito is not installed
    """
    if not DEVITO_AVAILABLE:
        raise ImportError("Devito is required for comparison")

    if frequencies is None:
        frequencies = np.array([5.0, 10.0, 15.0], dtype=np.float32)

    nfreq = len(frequencies)
    velocity = 1500.0
    extent = (500., 500.)
    f0 = 15.0

    # Grid setup
    grid = Grid(shape=(nx, ny), extent=extent)
    dx, dy = grid.spacing
    dt = 0.5 * float(min(dx, dy)) / velocity

    # Wavefield with full history saved
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4,
                     save=nt)

    # Slowness squared model
    m = Function(name='m', grid=grid)
    m.data[:] = 1.0 / velocity**2

    # Source setup
    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
    src.coordinates.data[:] = [[extent[0]/2, extent[1]/2]]
    time_values = np.arange(nt) * dt
    src.data[:, 0] = ricker_wavelet(time_values, f0)

    # Frequency dimension for on-the-fly DFT
    f = Dimension(name='f')
    freqs = Function(name='freqs', dimensions=(f,), shape=(nfreq,),
                     dtype=np.float32)
    freqs.data[:] = frequencies

    freq_modes_otf = Function(name='freq_modes_otf', dtype=np.complex64,
                              dimensions=(f, *grid.dimensions),
                              shape=(nfreq, *grid.shape))

    # Time dimension
    time_dim = grid.time_dim
    dt_spacing = time_dim.spacing

    # Vectorized Fourier basis
    omega = 2 * pi * freqs
    basis = exp(-1j * omega * time_dim * dt_spacing)

    # PDE: m * u_tt - laplacian(u) = 0
    pde = m * u.dt2 - u.laplace
    update = Eq(u.forward, solve(pde, u.forward))

    # Source injection: src * dt^2 / m
    src_term = src.inject(field=u.forward, expr=src * dt_spacing**2 / m)

    # DFT accumulation
    dft_eq = Inc(freq_modes_otf, basis * u)

    # Create and run operator with spacing map
    # Note: with save=nt, valid time indices are 0 to nt-1
    # The operator starts at time=2 (needs backward step) and runs to time_M
    op = Operator([update, src_term, dft_eq], subs=grid.spacing_map)
    op(time_M=nt-2, dt=dt)  # -2 because leapfrog needs backward access

    # Extract results
    otf_result = freq_modes_otf.data.copy()
    u_history = u.data.copy()

    # Compute reference DFT from full history
    ref_result = compute_reference_dft(u_history, frequencies, dt)

    # Compare
    errors = []
    for k in range(nfreq):
        # Relative error
        norm_ref = np.linalg.norm(ref_result[k])
        if norm_ref > 1e-10:
            rel_err = np.linalg.norm(otf_result[k] - ref_result[k]) / norm_ref
        else:
            rel_err = np.linalg.norm(otf_result[k] - ref_result[k])
        errors.append(rel_err)

    max_error = max(errors)
    passed = max_error < rtol

    details = {
        'frequencies': frequencies.tolist(),
        'errors': errors,
        'max_error': max_error,
        'rtol': rtol,
        'nx': nx, 'ny': ny, 'nt': nt,
        'dt': dt
    }

    return passed, max_error, details


def plot_fourier_modes(
    freq_modes: np.ndarray,
    frequencies: np.ndarray,
    save_path: str = None,
    vmax: float = None
):
    """
    Plot real and imaginary parts of Fourier modes.

    Parameters
    ----------
    freq_modes : ndarray
        Complex Fourier modes, shape (nfreq, nx, ny)
    frequencies : ndarray
        Frequency values in Hz
    save_path : str, optional
        Path to save figure
    vmax : float, optional
        Color scale limit (symmetric about zero)
    """
    import matplotlib.pyplot as plt

    nfreq = len(frequencies)

    if vmax is None:
        vmax = np.max(np.abs(freq_modes)) * 0.5

    fig, axes = plt.subplots(2, nfreq, figsize=(4*nfreq, 8))

    for i, f in enumerate(frequencies):
        # Real part
        im1 = axes[0, i].imshow(np.real(freq_modes[i]).T,
                                 cmap='seismic', origin='lower',
                                 vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f'{f:.0f} Hz (Real)')
        plt.colorbar(im1, ax=axes[0, i])

        # Imaginary part
        im2 = axes[1, i].imshow(np.imag(freq_modes[i]).T,
                                 cmap='seismic', origin='lower',
                                 vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f'{f:.0f} Hz (Imag)')
        plt.colorbar(im2, ax=axes[1, i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    print("On-the-fly DFT Examples")
    print("=" * 50)

    if not DEVITO_AVAILABLE:
        print("Devito not installed. Skipping Devito examples.")
        print("\nRicker wavelet test:")
        t = np.linspace(0, 0.5, 500)
        w = ricker_wavelet(t, f0=15.0)
        print(f"  Peak amplitude: {np.max(np.abs(w)):.4f}")
        print(f"  Peak time: {t[np.argmax(w)]:.4f} s")
    else:
        # Single frequency test
        print("\n1. Single frequency on-the-fly DFT")
        print("-" * 40)
        mode, info = run_otf_dft(nx=51, ny=51, nt=300, freq=10.0)
        print(f"Grid: {info['nx']}x{info['ny']}, {info['nt']} time steps")
        print(f"CFL: {info['cfl']:.3f}")
        print(f"Mode norm: {np.linalg.norm(mode):.2f}")

        # Multi-frequency test
        print("\n2. Multi-frequency on-the-fly DFT")
        print("-" * 40)
        modes, freqs, info = run_otf_dft_multifreq(
            nx=51, ny=51, nt=300,
            frequencies=np.array([5.0, 10.0, 15.0, 20.0])
        )
        print(f"Frequencies: {freqs}")
        print(f"Mode norms: {[f'{np.linalg.norm(modes[i]):.2f}' for i in range(len(freqs))]}")

        # Verification test
        print("\n3. Verification against reference DFT")
        print("-" * 40)
        passed, max_err, details = compare_otf_to_fft(nx=31, ny=31, nt=150)
        status = "PASSED" if passed else "FAILED"
        print(f"Status: {status}")
        print(f"Maximum relative error: {max_err:.2e}")
        print(f"Per-frequency errors: {[f'{e:.2e}' for e in details['errors']]}")

        # Plot results
        print("\n4. Generating plots...")
        plot_fourier_modes(modes, freqs, save_path="fourier_modes.png")
        print("Saved to fourier_modes.png")
