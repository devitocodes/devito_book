"""Source functions for electromagnetic simulations.

This module provides various source waveforms for exciting electromagnetic
fields in FDTD simulations, including:
    - Gaussian pulse (broadband excitation)
    - Sinusoidal (monochromatic)
    - Gaussian-modulated sinusoid (narrow-band)

All sources are designed to be smooth and have controlled bandwidth
to minimize numerical dispersion artifacts.

References:
    - Taflove & Hagness, Ch. 4: "Electromagnetic Wave Source Conditions"
"""

import numpy as np


def gaussian_pulse_em(
    t: np.ndarray,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a Gaussian pulse for electromagnetic excitation.

    The Gaussian pulse is useful for broadband excitation and
    transient analysis. It has zero DC content and smooth rise.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    t0 : float
        Center time (peak location) [s]
    sigma : float
        Temporal width (standard deviation) [s]
    amplitude : float
        Peak amplitude [V/m or A/m]

    Returns
    -------
    np.ndarray
        Gaussian pulse values at times t

    Notes
    -----
    The pulse is defined as:
        g(t) = A * exp(-((t - t0) / sigma)^2)

    The -3dB bandwidth is approximately 0.265 / sigma.

    Examples
    --------
    >>> t = np.linspace(0, 1e-8, 1000)
    >>> pulse = gaussian_pulse_em(t, t0=5e-9, sigma=1e-9)
    """
    return amplitude * np.exp(-((t - t0) / sigma) ** 2)


def sinusoidal_source(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    t_ramp: float | None = None,
) -> np.ndarray:
    """Generate a sinusoidal (CW) source with optional soft turn-on.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Frequency [Hz]
    amplitude : float
        Peak amplitude [V/m or A/m]
    phase : float
        Initial phase [radians]
    t_ramp : float, optional
        Ramp-up time for soft turn-on [s].
        If None, no ramping is applied.

    Returns
    -------
    np.ndarray
        Sinusoidal source values

    Notes
    -----
    The soft turn-on ramp uses a raised cosine function to
    avoid high-frequency content from an abrupt start:
        ramp(t) = 0.5 * (1 - cos(pi * t / t_ramp)) for t < t_ramp

    Examples
    --------
    >>> t = np.linspace(0, 1e-8, 1000)
    >>> src = sinusoidal_source(t, f0=1e9, t_ramp=2e-9)
    """
    omega = 2.0 * np.pi * f0
    signal = amplitude * np.sin(omega * t + phase)

    if t_ramp is not None:
        ramp = np.where(
            t < t_ramp,
            0.5 * (1.0 - np.cos(np.pi * t / t_ramp)),
            1.0,
        )
        signal = signal * ramp

    return signal


def gaussian_modulated_source(
    t: np.ndarray,
    f0: float,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a Gaussian-modulated sinusoidal pulse.

    This creates a narrow-band pulse centered at frequency f0,
    useful for studying resonances and frequency-selective behavior.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Center frequency [Hz]
    t0 : float
        Center time (envelope peak) [s]
    sigma : float
        Temporal width of Gaussian envelope [s]
    amplitude : float
        Peak amplitude [V/m or A/m]

    Returns
    -------
    np.ndarray
        Gaussian-modulated sinusoidal values

    Notes
    -----
    The waveform is:
        s(t) = A * sin(2*pi*f0*t) * exp(-((t - t0) / sigma)^2)

    The spectrum is a Gaussian centered at f0 with bandwidth
    proportional to 1/sigma.

    Examples
    --------
    >>> t = np.linspace(0, 1e-8, 1000)
    >>> pulse = gaussian_modulated_source(t, f0=5e9, t0=5e-9, sigma=1e-9)
    """
    envelope = np.exp(-((t - t0) / sigma) ** 2)
    carrier = np.sin(2.0 * np.pi * f0 * t)
    return amplitude * envelope * carrier


def differentiated_gaussian(
    t: np.ndarray,
    t0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a differentiated Gaussian pulse.

    Also known as a "first derivative Gaussian" or "Gaussian monocycle",
    this pulse has zero DC content, making it suitable for antenna
    simulations where DC must be avoided.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    t0 : float
        Center time [s]
    sigma : float
        Temporal width [s]
    amplitude : float
        Amplitude scaling factor

    Returns
    -------
    np.ndarray
        Differentiated Gaussian values

    Notes
    -----
    The waveform is proportional to:
        d/dt[exp(-((t-t0)/sigma)^2)] = -2*(t-t0)/sigma^2 * exp(-((t-t0)/sigma)^2)
    """
    tau = (t - t0) / sigma
    return -amplitude * 2.0 * tau * np.exp(-tau**2)


def ricker_wavelet_em(
    t: np.ndarray,
    f0: float,
    t0: float | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat) for EM simulations.

    The Ricker wavelet is the negative normalized second derivative
    of a Gaussian, with zero mean and zero DC content.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Peak frequency [Hz]
    t0 : float, optional
        Time delay [s]. Default: 1/f0 (one period delay)
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        Ricker wavelet values

    Notes
    -----
    The Ricker wavelet is defined as:
        R(t) = A * (1 - 2*pi^2*f0^2*(t-t0)^2) * exp(-pi^2*f0^2*(t-t0)^2)

    The peak frequency is f0, and the bandwidth is approximately 1.2*f0.
    """
    if t0 is None:
        t0 = 1.0 / f0

    pi_f0_tau = np.pi * f0 * (t - t0)
    return amplitude * (1.0 - 2.0 * pi_f0_tau**2) * np.exp(-pi_f0_tau**2)


def plane_wave_tf_sf_1d(
    t: np.ndarray,
    f0: float,
    n_periods: int = 3,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a truncated sinusoidal waveform for TF/SF excitation.

    Total-Field/Scattered-Field (TF/SF) is a technique to inject
    plane waves into FDTD simulations. This function generates the
    incident field waveform with smooth turn-on and turn-off.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    f0 : float
        Frequency [Hz]
    n_periods : int
        Number of complete periods before turn-off
    amplitude : float
        Peak amplitude

    Returns
    -------
    np.ndarray
        TF/SF excitation waveform

    Notes
    -----
    The waveform includes:
        1. Half-period raised cosine turn-on
        2. n_periods of full sinusoid
        3. Half-period raised cosine turn-off
    """
    period = 1.0 / f0
    omega = 2.0 * np.pi * f0

    # Time markers
    t_on = period / 2  # Turn-on duration
    t_full_end = t_on + n_periods * period  # End of full-amplitude region
    t_total = t_full_end + period / 2  # Total active duration

    signal = np.zeros_like(t)

    # Turn-on phase (raised cosine ramp)
    mask_on = (t >= 0) & (t < t_on)
    ramp_on = 0.5 * (1.0 - np.cos(np.pi * t[mask_on] / t_on))
    signal[mask_on] = ramp_on * np.sin(omega * t[mask_on])

    # Full amplitude phase
    mask_full = (t >= t_on) & (t < t_full_end)
    signal[mask_full] = np.sin(omega * t[mask_full])

    # Turn-off phase (raised cosine ramp down)
    mask_off = (t >= t_full_end) & (t < t_total)
    t_local = t[mask_off] - t_full_end
    ramp_off = 0.5 * (1.0 + np.cos(np.pi * t_local / (period / 2)))
    signal[mask_off] = ramp_off * np.sin(omega * t[mask_off])

    return amplitude * signal
