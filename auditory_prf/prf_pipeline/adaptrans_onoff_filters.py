import numpy as np
from scipy.signal import decimate
#19/02/2026
# Adapted from: https://github.com/urancon/deepSTRF/blob/9be7ca5698ab856990458834af8a2e412480823e/deepSTRF/models/prefiltering.py
# deepSTRF/models/prefiltering.py

def downsample_AN(an_output: np.ndarray, factor: int) -> np.ndarray:
    """
    Anti-aliased downsampling of AN output along time axis.

    Parameters
    ----------
    an_output : np.ndarray, shape (N_CFs, T)
        One combined channel per CF (HSR/MSR/LSR already merged).
    factor : int
        Downsampling factor.
        e.g. dt=0.1ms, factor=100 → 10ms bins.

    Returns
    -------
    downsampled : np.ndarray, shape (N_CFs, T // factor)
    """
    return np.stack([
        decimate(an_output[cf], factor, ftype='fir', zero_phase=True)
        for cf in range(an_output.shape[0])
    ])


def tau_to_a(tau_ms: float, dt_ms: float) -> float:
    """Convert a time constant (ms) to exponential decay rate 'a'."""
    return np.exp(-dt_ms / tau_ms)


def willmore_tau(cf_hz: float) -> float:
    """
    Frequency-dependent time constant from Willmore et al. (2016).
    tau(f) = 500 - 105 * log10(f_Hz)  [ms]
    """
    return 500.0 - 105.0 * np.log10(cf_hz)


def build_ON_kernel(a: float, w: float, K: int) -> np.ndarray:
    """
    FIR onset kernel for a single CF channel.

    Shape: [-C*w, -C*w*a, ..., -C*w*a^(K-2), +1]
    Detects increases relative to exponential average of recent past.

    Parameters
    ----------
    a : float in (0, 1)
        Exponential decay rate. a = exp(-dt / tau)
    w : float in (0, 1)
        Adaptation weight. Higher = stronger subtraction of past.
    K : int
        Kernel length in samples.
    """
    exponents = np.arange(K - 1)
    exp_terms = a ** exponents        # [1, a, a^2, ..., a^(K-2)]
    C = 1.0 / exp_terms.sum()         # normalization

    kernel = np.empty(K)
    kernel[-1] = 1.0                   # current sample: +1
    kernel[:-1] = -C * w * exp_terms  # past samples:   -C*w*a^i
    return kernel


def build_OFF_kernel(a: float, w: float, K: int) -> np.ndarray:
    """
    FIR offset kernel for a single CF channel.

    Derived from ON kernel: kernel_OFF = -kernel_ON / w, last tap = -w
    Detects decreases relative to exponential average of recent past.

    Parameters
    ----------
    a : float in (0, 1)
    w : float in (0, 1)
    K : int
    """
    on_kernel = build_ON_kernel(a, w, K)
    off_kernel = -on_kernel / w
    off_kernel[-1] = -w
    return off_kernel


def apply_adaptrans(an_output: np.ndarray,
                    CFs_Hz: np.ndarray,
                    dt_ms: float,
                    w: float = 0.8,
                    K: int = None,
                    rectify: bool = True) -> np.ndarray:
    """
    Apply AdapTrans ON/OFF filters to downsampled AN output.

    Parameters
    ----------
    an_output : np.ndarray, shape (N_CFs, T)
        Downsampled AN output, one channel per CF.
    CFs_Hz : np.ndarray, shape (N_CFs,)
        Characteristic frequency of each channel in Hz.
    dt_ms : float
        Time step of the downsampled signal in milliseconds.
    w : float
        Adaptation weight, same for all CFs. Default 0.8.
        Will become a learnable parameter during model fitting later.
    K : int or None
        Kernel length in samples. If None, auto-set to cover
        3x the longest time constant across all CFs.
    rectify : bool
        Half-wave rectify output (ReLU). Default True.

    Returns
    -------
    out : np.ndarray, shape (2, N_CFs, T)
        out[0] = ON  (onset)  responses
        out[1] = OFF (offset) responses
    """
    N_CFs, T = an_output.shape

    # per-CF time constants and decay rates from Willmore et al.
    tau_vals = np.array([willmore_tau(cf) for cf in CFs_Hz])      # (N_CFs,) ms
    a_vals   = np.array([tau_to_a(tau, dt_ms) for tau in tau_vals]) # (N_CFs,)

    # auto-set K to cover 3x the longest time constant if not specified
    if K is None:
        max_tau_samples = np.max(tau_vals) / dt_ms        # time constant in samples
        K = int(np.ceil(3 * max_tau_samples))             # cover 3x the longest tau
        print(f"Auto-set K={K} samples "
            f"(3 × max tau={np.max(tau_vals):.1f}ms / dt={dt_ms}ms)")

    out_ON  = np.zeros((N_CFs, T))
    out_OFF = np.zeros((N_CFs, T))

    for i in range(N_CFs):
        kernel_ON  = build_ON_kernel(a_vals[i], w, K)
        kernel_OFF = build_OFF_kernel(a_vals[i], w, K)

        # causal padding: replicate first sample to avoid onset artifact
        signal = an_output[i]
        padded = np.concatenate([np.full(K - 1, signal[0]), signal])

        out_ON[i]  = np.convolve(padded, kernel_ON[::-1],  mode='valid')[:T]
        out_OFF[i] = np.convolve(padded, kernel_OFF[::-1], mode='valid')[:T]

    if rectify:
        out_ON  = np.maximum(out_ON,  0.0)
        out_OFF = np.maximum(out_OFF, 0.0)

    return np.stack([out_ON, out_OFF], axis=0)  # (2, N_CFs, T)


def preprocess_AN_output(an_output: np.ndarray,
                         CFs_Hz: np.ndarray,
                         dt_fine_ms: float,
                         downsample_factor: int,
                         w: float = 0.8,
                         K: int = None) -> np.ndarray:
    """
    Full preprocessing pipeline: AN output → ON/OFF representation.

    Parameters
    ----------
    an_output : np.ndarray, shape (N_CFs, T_fine)
        AN model output, one combined channel per CF.
    CFs_Hz : np.ndarray, shape (N_CFs,)
        Characteristic frequencies in Hz.
    dt_fine_ms : float
        Time step of the raw AN output in ms. e.g. 0.1ms
    downsample_factor : int
        Downsampling factor. e.g. 100 → 10ms bins.
    w : float
        Adaptation weight. Default 0.8.
    K : int or None
        Kernel length in samples. Auto-set if None.

    Returns
    -------
    on_off : np.ndarray, shape (2, N_CFs, T_coarse)
        ON/OFF filtered AN output, ready for encoding model.
    """
    dt_coarse_ms = dt_fine_ms * downsample_factor

    downsampled = downsample_AN(an_output, downsample_factor)  # (N_CFs, T_coarse)
    on_off      = apply_adaptrans(downsampled, CFs_Hz,
                                  dt_coarse_ms, w=w, K=K)      # (2, N_CFs, T_coarse)
    return on_off