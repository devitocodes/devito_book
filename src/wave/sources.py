"""Source wavelet functions for seismic and wave propagation modeling.

This module provides common source wavelets used in seismic imaging
and wave propagation simulations:
- Ricker wavelet (Mexican hat wavelet)
- Gaussian pulse
- Derivative of Gaussian

These are typically used with Devito's SparseTimeFunction for source injection.

Usage:
    from src.wave import ricker_wavelet, gaussian_pulse

    # Create time array
    t = np.linspace(0, 1, 1001)

    # Generate Ricker wavelet with 10 Hz peak frequency
    src = ricker_wavelet(t, f0=10.0, t0=0.1)

    # Generate Gaussian pulse
    src = gaussian_pulse(t, t0=0.1, sigma=0.02)
"""

import numpy as np


def ricker_wavelet(
    t: np.ndarray,
    f0: float = 10.0,
    t0: float | None = None,
    amp: float = 1.0,
) -> np.ndarray:
    """Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the negative normalized second derivative of a
    Gaussian. It's commonly used in seismic modeling due to its compact
    support in both time and frequency domains.

    r(t) = amp * (1 - 2*(pi*f0*(t-t0))^2) * exp(-(pi*f0*(t-t0))^2)

    Parameters
    ----------
    t : np.ndarray
        Time array
    f0 : float
        Peak frequency in Hz (dominant frequency)
    t0 : float, optional
        Time shift (delay). If None, defaults to 1.5/f0 to avoid
        negative time values for the wavelet center.
    amp : float
        Amplitude scaling factor

    Returns
    -------
    np.ndarray
        Ricker wavelet values at times t

    Notes
    -----
    The wavelet has zero mean and is bandlimited. The frequency spectrum
    has a peak at f0 and falls off on both sides. The wavelet is
    essentially zero outside |t - t0| > 1/f0.

    Examples
    --------
    >>> t = np.linspace(0, 0.5, 501)
    >>> src = ricker_wavelet(t, f0=25.0)
    >>> plt.plot(t, src)
    """
    if t0 is None:
        t0 = 1.5 / f0  # Delay so wavelet starts near zero

    # Normalized time
    tau = np.pi * f0 * (t - t0)
    tau_sq = tau ** 2

    return amp * (1.0 - 2.0 * tau_sq) * np.exp(-tau_sq)


def gaussian_pulse(
    t: np.ndarray,
    t0: float = 0.1,
    sigma: float = 0.02,
    amp: float = 1.0,
) -> np.ndarray:
    """Generate a Gaussian pulse.

    g(t) = amp * exp(-((t - t0) / sigma)^2 / 2)

    Parameters
    ----------
    t : np.ndarray
        Time array
    t0 : float
        Center time of the pulse
    sigma : float
        Standard deviation (controls pulse width)
    amp : float
        Amplitude

    Returns
    -------
    np.ndarray
        Gaussian pulse values at times t

    Notes
    -----
    The Gaussian pulse is infinitely smooth and has good frequency
    localization. However, it has non-zero DC component.

    Examples
    --------
    >>> t = np.linspace(0, 0.5, 501)
    >>> src = gaussian_pulse(t, t0=0.1, sigma=0.02)
    """
    return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def gaussian_derivative(
    t: np.ndarray,
    t0: float = 0.1,
    sigma: float = 0.02,
    amp: float = 1.0,
) -> np.ndarray:
    """Generate first derivative of Gaussian pulse.

    g'(t) = -amp * (t - t0) / sigma^2 * exp(-((t - t0) / sigma)^2 / 2)

    Parameters
    ----------
    t : np.ndarray
        Time array
    t0 : float
        Center time
    sigma : float
        Standard deviation
    amp : float
        Amplitude

    Returns
    -------
    np.ndarray
        Derivative of Gaussian values at times t

    Notes
    -----
    This wavelet has zero mean (no DC component) and is commonly
    used when a zero-mean source is needed.
    """
    tau = (t - t0) / sigma
    return -amp * tau / sigma * np.exp(-0.5 * tau ** 2)


def sinc_wavelet(
    t: np.ndarray,
    f_max: float = 50.0,
    t0: float | None = None,
    amp: float = 1.0,
) -> np.ndarray:
    """Generate a sinc wavelet (bandlimited impulse).

    s(t) = amp * sin(2*pi*f_max*(t-t0)) / (pi*(t-t0))

    Parameters
    ----------
    t : np.ndarray
        Time array
    f_max : float
        Maximum frequency (cutoff frequency)
    t0 : float, optional
        Time shift. Default: center of time array
    amp : float
        Amplitude

    Returns
    -------
    np.ndarray
        Sinc wavelet values

    Notes
    -----
    The sinc function is the ideal lowpass filter impulse response.
    It contains all frequencies up to f_max with equal amplitude.
    """
    if t0 is None:
        t0 = (t[0] + t[-1]) / 2

    # Avoid division by zero at t = t0
    tau = t - t0
    result = np.zeros_like(t)

    # Handle t = t0 case
    mask = np.abs(tau) > 1e-15
    result[mask] = amp * np.sin(2 * np.pi * f_max * tau[mask]) / (np.pi * tau[mask])
    result[~mask] = amp * 2 * f_max  # Limit as tau -> 0

    return result


def get_source_spectrum(
    wavelet: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency spectrum of a source wavelet.

    Parameters
    ----------
    wavelet : np.ndarray
        Time-domain wavelet
    dt : float
        Time step

    Returns
    -------
    tuple
        (frequencies, amplitude_spectrum)
        frequencies in Hz, amplitude is normalized

    Examples
    --------
    >>> t = np.linspace(0, 1, 1001)
    >>> dt = t[1] - t[0]
    >>> src = ricker_wavelet(t, f0=10.0)
    >>> freq, amp = get_source_spectrum(src, dt)
    >>> plt.plot(freq, amp)
    """
    n = len(wavelet)
    spectrum = np.fft.rfft(wavelet)
    frequencies = np.fft.rfftfreq(n, dt)
    amplitude = np.abs(spectrum) / n

    return frequencies, amplitude


def estimate_peak_frequency(
    wavelet: np.ndarray,
    dt: float,
) -> float:
    """Estimate the peak frequency of a source wavelet.

    Parameters
    ----------
    wavelet : np.ndarray
        Time-domain wavelet
    dt : float
        Time step

    Returns
    -------
    float
        Estimated peak frequency in Hz
    """
    freq, amp = get_source_spectrum(wavelet, dt)
    idx_peak = np.argmax(amp)
    return freq[idx_peak]
