"""Ground Penetrating Radar (GPR) simulation utilities.

Provides tools for simulating GPR surveys using FDTD, including:
- Ricker wavelet and other source wavelets
- B-scan generation (2D radargram)
- Material models for soil
- Buried target scenarios

GPR operates by transmitting EM pulses into the ground and recording
reflections from subsurface interfaces and objects. Typical frequencies
range from 100 MHz to 2 GHz.

References
----------
.. [1] C. Warren et al., "gprMax: Open source software to simulate
       electromagnetic wave propagation for Ground Penetrating Radar,"
       Computer Physics Communications, vol. 209, pp. 163-170, 2016.

.. [2] D.J. Daniels, "Ground Penetrating Radar," 2nd ed., IET, 2004.
"""

from dataclasses import dataclass

import numpy as np

from src.em.materials import DielectricMaterial
from src.em.maxwell1D_devito import solve_maxwell_1d
from src.em.maxwell2D_devito import solve_maxwell_2d
from src.em.units import EMConstants


def ricker_wavelet(
    t: np.ndarray,
    f0: float,
    t0: float = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Ricker wavelet (Mexican hat) source.

    The Ricker wavelet is the negative normalized second derivative of
    a Gaussian, commonly used in GPR and seismic simulations.

    r(t) = A * (1 - 2*(pi*f0*(t-t0))^2) * exp(-(pi*f0*(t-t0))^2)

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Peak (dominant) frequency [Hz]
    t0 : float, optional
        Time delay [s]. Default: 1/f0 (one period)
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Wavelet amplitude at each time
    """
    if t0 is None:
        t0 = 1.0 / f0

    tau = np.pi * f0 * (t - t0)
    return amplitude * (1 - 2 * tau**2) * np.exp(-tau**2)


def gaussian_derivative_wavelet(
    t: np.ndarray,
    f0: float,
    t0: float = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """First derivative of Gaussian wavelet.

    Also known as the Gaussian monocycle, this wavelet has a broader
    bandwidth than the Ricker wavelet.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Characteristic frequency [Hz]
    t0 : float, optional
        Time delay [s]
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Wavelet amplitude
    """
    if t0 is None:
        t0 = 1.0 / f0

    tau = (t - t0) * f0
    sigma = 1 / (2 * np.pi * f0)
    return amplitude * (-tau / sigma**2) * np.exp(-0.5 * (tau / sigma)**2)


def blackman_harris_wavelet(
    t: np.ndarray,
    f0: float,
    t0: float = None,
    n_cycles: int = 4,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Blackman-Harris windowed sinusoid.

    Provides better spectral characteristics with reduced side lobes.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Center frequency [Hz]
    t0 : float, optional
        Time delay [s]
    n_cycles : int
        Number of cycles in the pulse
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Wavelet amplitude
    """
    if t0 is None:
        t0 = n_cycles / (2 * f0)

    T = n_cycles / f0  # Total duration
    t_rel = t - t0 + T/2

    # Blackman-Harris window
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    window = np.zeros_like(t)
    mask = (t_rel >= 0) & (t_rel <= T)
    tau = t_rel[mask] / T
    window[mask] = (a0 - a1*np.cos(2*np.pi*tau) +
                   a2*np.cos(4*np.pi*tau) - a3*np.cos(6*np.pi*tau))

    # Modulated sinusoid
    return amplitude * window * np.sin(2 * np.pi * f0 * (t - t0))


def wavelet_spectrum(
    wavelet: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute amplitude spectrum of a wavelet.

    Parameters
    ----------
    wavelet : np.ndarray
        Time-domain wavelet
    dt : float
        Time step [s]

    Returns
    -------
    tuple
        (frequencies [Hz], amplitude spectrum)
    """
    n = len(wavelet)
    freq = np.fft.rfftfreq(n, dt)
    spectrum = np.abs(np.fft.rfft(wavelet))
    return freq, spectrum


@dataclass
class GPRResult:
    """Results from GPR simulation.

    Attributes
    ----------
    ascan : np.ndarray
        A-scan (single trace) time series
    t : np.ndarray
        Time array [s]
    x : np.ndarray
        Spatial coordinate (depth for 1D, surface position for B-scan)
    bscan : np.ndarray, optional
        B-scan (2D radargram) if multiple traces recorded
    positions : np.ndarray, optional
        Antenna positions for B-scan
    depth_axis : np.ndarray, optional
        Converted depth axis (using estimated velocity)
    """
    ascan: np.ndarray
    t: np.ndarray
    x: np.ndarray
    bscan: np.ndarray | None = None
    positions: np.ndarray | None = None
    depth_axis: np.ndarray | None = None


def two_way_travel_time(depth: float, eps_r: float) -> float:
    """Compute two-way travel time for a target at given depth.

    Parameters
    ----------
    depth : float
        Target depth [m]
    eps_r : float
        Relative permittivity of medium

    Returns
    -------
    float
        Two-way travel time [s]
    """
    const = EMConstants()
    v = const.c0 / np.sqrt(eps_r)
    return 2 * depth / v


def depth_from_travel_time(twtt: float, eps_r: float) -> float:
    """Convert two-way travel time to depth.

    Parameters
    ----------
    twtt : float
        Two-way travel time [s]
    eps_r : float
        Relative permittivity of medium

    Returns
    -------
    float
        Depth [m]
    """
    const = EMConstants()
    v = const.c0 / np.sqrt(eps_r)
    return twtt * v / 2


def run_gpr_1d(
    depth: float,
    eps_r_soil: float,
    sigma_soil: float,
    frequency: float = 500e6,
    target_depth: float | None = None,
    target_eps_r: float = 1.0,
    time_window: float = None,
    Nx: int = 400,
) -> GPRResult:
    """Run 1D GPR simulation (vertical profile).

    Parameters
    ----------
    depth : float
        Total simulation depth [m]
    eps_r_soil : float
        Soil relative permittivity
    sigma_soil : float
        Soil conductivity [S/m]
    frequency : float
        Center frequency [Hz]
    target_depth : float, optional
        Depth of reflector/target [m]
    target_eps_r : float
        Target relative permittivity
    time_window : float, optional
        Recording time [s]. Default: auto-computed.
    Nx : int
        Number of grid points

    Returns
    -------
    GPRResult
        Simulation results including A-scan
    """
    # Compute time window if not specified
    const = EMConstants()
    v_soil = const.c0 / np.sqrt(eps_r_soil)

    if time_window is None:
        time_window = 4 * depth / v_soil

    # Grid spacing (10 points per wavelength in soil)
    wavelength = v_soil / frequency
    dx = wavelength / 20
    Nx = max(Nx, int(depth / dx) + 1)

    # Material arrays
    x = np.linspace(0, depth, Nx + 1)
    eps_r = np.full(Nx + 1, eps_r_soil)
    sigma = np.full(Nx + 1, sigma_soil)

    # Add target layer if specified
    if target_depth is not None:
        target_mask = x >= target_depth
        eps_r[target_mask] = target_eps_r
        sigma[target_mask] = 0.0  # Assume lossless target

    # Source function (Ricker wavelet)
    def source(t):
        return ricker_wavelet(np.array([t]), frequency)[0]

    # Run simulation
    result = solve_maxwell_1d(
        L=depth,
        Nx=Nx,
        T=time_window,
        CFL=0.9,
        eps_r=eps_r,
        sigma=sigma,
        source_func=source,
        source_position=dx,  # Near top
        bc_left="abc",
        bc_right="abc",
        save_history=True,
    )

    # Extract A-scan (field at source position)
    src_idx = 1
    ascan = result.E_history[:, src_idx] if result.E_history is not None else result.E_z

    # Convert time to depth axis
    depth_axis = result.t_history * v_soil / 2 if result.t_history is not None else None

    return GPRResult(
        ascan=ascan,
        t=result.t_history if result.t_history is not None else np.array([result.t]),
        x=result.x_E,
        depth_axis=depth_axis,
    )


def run_gpr_bscan_2d(
    Lx: float,
    Ly: float,
    eps_r_background: float,
    sigma_background: float,
    frequency: float = 500e6,
    n_traces: int = 50,
    target_center: tuple | None = None,
    target_radius: float = 0.05,
    target_material: DielectricMaterial | None = None,
    time_window: float = None,
    Nx: int = 200,
    Ny: int = 200,
) -> GPRResult:
    """Run 2D GPR B-scan simulation.

    Simulates a GPR survey line with the antenna moving along the
    surface (x-direction) and recording reflections from below (y-direction).

    Parameters
    ----------
    Lx : float
        Survey line length [m]
    Ly : float
        Survey depth [m]
    eps_r_background : float
        Background (soil) relative permittivity
    sigma_background : float
        Background conductivity [S/m]
    frequency : float
        Center frequency [Hz]
    n_traces : int
        Number of traces (antenna positions)
    target_center : tuple, optional
        (x, y) center of buried target [m]
    target_radius : float
        Target radius [m]
    target_material : DielectricMaterial, optional
        Target material. Default: PEC-like.
    time_window : float, optional
        Recording time per trace [s]
    Nx, Ny : int
        Grid points in x and y

    Returns
    -------
    GPRResult
        B-scan radargram and trace data
    """
    const = EMConstants()
    v = const.c0 / np.sqrt(eps_r_background)

    if time_window is None:
        time_window = 3 * Ly / v

    # Antenna positions along survey line
    positions = np.linspace(0.1 * Lx, 0.9 * Lx, n_traces)

    # For a full B-scan, we'd run n_traces simulations
    # Here we provide a simplified version that runs one simulation
    # and extracts traces at different x-positions

    # Create material model
    from src.em.materials import DielectricMaterial, create_cylinder_model_2d

    background = DielectricMaterial(
        name="Background",
        eps_r=eps_r_background,
        sigma=sigma_background,
    )

    if target_center is not None:
        if target_material is None:
            target_material = DielectricMaterial(
                name="Target", eps_r=1.0, sigma=1e6  # PEC-like
            )

        eps_r, sigma = create_cylinder_model_2d(
            Nx, Ny, Lx, Ly,
            target_center, target_radius,
            target_material, background
        )
    else:
        eps_r = np.full((Nx + 1, Ny + 1), eps_r_background)
        sigma = np.full((Nx + 1, Ny + 1), sigma_background)

    # For demonstration, run single simulation with central source
    # Full B-scan would loop over antenna positions
    x_src = Lx / 2
    y_src = 0.02  # Just below surface

    def source(t):
        return ricker_wavelet(np.array([t]), frequency)[0]

    result = solve_maxwell_2d(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        T=time_window,
        CFL=0.5,
        eps_r=eps_r,
        sigma=sigma,
        source_func=source,
        source_position=(x_src, y_src),
        pml_width=15,
        save_history=True,
        save_every=5,
    )

    # Extract B-scan from history
    # Each column of B-scan is the field along a vertical line at different times
    if result.E_history is not None and len(result.E_history) > 0:
        # Get field at surface (y=0) for all x positions over time
        bscan = np.zeros((len(result.E_history), n_traces))
        x_indices = (positions / Lx * Nx).astype(int)
        x_indices = np.clip(x_indices, 0, Nx)

        for i, E in enumerate(result.E_history):
            bscan[i, :] = E[x_indices, 1]  # y index 1 is just below surface
    else:
        bscan = None

    # Extract single A-scan at center position
    center_idx = n_traces // 2
    ascan = bscan[:, center_idx] if bscan is not None else result.E_z[Nx//2, :]

    return GPRResult(
        ascan=ascan,
        t=result.t_history if result.t_history is not None else np.array([result.t]),
        x=result.x,
        bscan=bscan,
        positions=positions,
        depth_axis=result.t_history * v / 2 if result.t_history is not None else None,
    )


def hyperbola_travel_time(
    x_antenna: float,
    x_target: float,
    y_target: float,
    v: float,
) -> float:
    """Compute travel time for hyperbolic diffraction.

    The travel time from a point scatterer creates a hyperbolic
    pattern in a B-scan radargram.

    Parameters
    ----------
    x_antenna : float
        Antenna position along survey line [m]
    x_target : float
        Target x-position [m]
    y_target : float
        Target depth [m]
    v : float
        Wave velocity in medium [m/s]

    Returns
    -------
    float
        Two-way travel time [s]
    """
    distance = np.sqrt((x_antenna - x_target)**2 + y_target**2)
    return 2 * distance / v


def fit_hyperbola(
    x_positions: np.ndarray,
    travel_times: np.ndarray,
) -> tuple[float, float, float]:
    """Fit hyperbola to diffraction curve to estimate velocity and depth.

    The hyperbolic equation is:
    t(x) = (2/v) * sqrt((x - x0)^2 + z0^2)

    Parameters
    ----------
    x_positions : np.ndarray
        Antenna positions [m]
    travel_times : np.ndarray
        Observed travel times [s]

    Returns
    -------
    tuple
        (x0: target x-position [m], z0: target depth [m], v: velocity [m/s])
    """
    from scipy.optimize import curve_fit

    def hyperbola(x, x0, z0, v):
        return (2/v) * np.sqrt((x - x0)**2 + z0**2)

    # Initial guess
    x0_init = x_positions[np.argmin(travel_times)]
    t_min = np.min(travel_times)
    v_init = 0.1 * EMConstants().c0  # Assume soil-like velocity
    z0_init = t_min * v_init / 2

    popt, _ = curve_fit(
        hyperbola, x_positions, travel_times,
        p0=[x0_init, z0_init, v_init],
        bounds=([x_positions.min(), 0, 1e6], [x_positions.max(), 10, 3e8])
    )

    return popt[0], popt[1], popt[2]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GPRResult",
    "blackman_harris_wavelet",
    "depth_from_travel_time",
    "fit_hyperbola",
    "gaussian_derivative_wavelet",
    "hyperbola_travel_time",
    "ricker_wavelet",
    "run_gpr_1d",
    "run_gpr_bscan_2d",
    "two_way_travel_time",
    "wavelet_spectrum",
]
