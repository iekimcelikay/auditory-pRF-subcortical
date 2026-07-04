import logging
from typing import Optional

import numpy as np
from scipy.signal import decimate
#19/02/2026
# Adapted from: https://github.com/urancon/deepSTRF/blob/9be7ca5698ab856990458834af8a2e412480823e/deepSTRF/models/prefiltering.py
# deepSTRF/models/prefiltering.py

logger = logging.getLogger(__name__)


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


def cf_to_tau_ms(cf_hz: float, cf_range_hz: tuple, tau_range_ms: tuple = (10.0, 500.0)) -> float:
    """Map a CF (Hz) to an AdapTrans time constant (ms), log-linear in CF.

    Same decreasing-with-frequency shape as Willmore et al. (2016), but
    re-anchored so the lowest CF in ``cf_range_hz`` maps to
    ``tau_range_ms[1]`` (longest tau) and the highest CF maps to
    ``tau_range_ms[0]`` (shortest tau).

    Parameters
    ----------
    cf_hz : float
        Characteristic frequency in Hz.
    cf_range_hz : tuple of (min_hz, max_hz)
        CF range spanned by the experiment.
    tau_range_ms : tuple of (tau_min_ms, tau_max_ms)
        Time-constant range. Default (10.0, 500.0).

    Returns
    -------
    tau_ms : float
    """
    cf_min_hz, cf_max_hz = cf_range_hz
    tau_min_ms, tau_max_ms = tau_range_ms
    if cf_max_hz == cf_min_hz:
        return float(np.mean(tau_range_ms))
    cf_clamped = np.clip(cf_hz, cf_min_hz, cf_max_hz)
    frac = ((np.log10(cf_clamped) - np.log10(cf_min_hz))
            / (np.log10(cf_max_hz) - np.log10(cf_min_hz)))  # 0 at low CF, 1 at high CF
    return tau_max_ms - frac * (tau_max_ms - tau_min_ms)


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
    kernel[0] = 1.0                   # current sample: +1
    kernel[1:] = -C * w * exp_terms  # past samples:   -C*w*a^i
    return kernel


def build_OFF_kernel(a: float, w: float, K: int) -> np.ndarray:
    """
    FIR offset kernel for a single CF channel.

    h_OFF[0]   = -w                        (current sample, discounted)
    h_OFF[d]   = +C * a^(d-1)   d=1..K-1  (exponentially weighted past)

    This is intentionally NOT the exact negative of h_ON. The asymmetry is
    by design: ON discounts the *past* by w, OFF discounts the *present* by w.
    See docs/pipeline_equations.md for the full derivation.

    Algebraic shortcut used below:
      -h_ON / w  gives taps d>=1:  -(-C*w*a^(d-1)) / w = +C*a^(d-1)  (correct)
                 but   tap 0:      -(+1) / w            = -1/w         (wrong)
      So tap 0 is overwritten with -w.

    Parameters
    ----------
    a : float in (0, 1)
        Exponential decay rate. a = exp(-dt / tau)
    w : float in (0, 1)
        Adaptation weight. Higher = stronger subtraction of present.
    K : int
        Kernel length in samples.
    """
    on_kernel = build_ON_kernel(a, w, K)
    off_kernel = -on_kernel / w
    off_kernel[0] = -w
    return off_kernel


def apply_adaptrans(an_output: np.ndarray,
                    CFs_Hz: np.ndarray,
                    dt_ms: float,
                    w: float = 0.8,
                    K: Optional[int] = None,
                    rectify: bool = False,
                    pad_value: Optional[float] = None,
                    tau_ms: float = 100.0,
                    tau_ms_off: Optional[float] = None) -> np.ndarray:
    """
    Apply AdapTrans ON/OFF filters to downsampled AN output.

    tau_ms and tau_ms_off are free parameters with no assumed relationship to CF.

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
    K : int or None
        Kernel length in samples. If None, auto-set to cover 3x the longer tau.
    rectify : bool
        Half-wave rectify output (ReLU). Default False.
    pad_value : float or None
        Value used to pad the left edge of each channel before convolution.
        If None, replicates signal[0] (standard causal padding). Pass 0.0
        for boxcar trains that start with non-zero amplitude at t=0.
    tau_ms : float
        ON-filter time constant (ms). Default 100.0.
        Free parameter — no CF-tau relationship assumed.
    tau_ms_off : float or None
        OFF-filter time constant (ms). If None, uses tau_ms for both ON and OFF.

    Returns
    -------
    out : np.ndarray, shape (2, N_CFs, T)
        out[0] = ON  (onset)  responses
        out[1] = OFF (offset) responses
    """
    N_CFs, T = an_output.shape

    tau_vals_on  = np.full(N_CFs, float(tau_ms))
    tau_vals_off = np.full(N_CFs, float(tau_ms_off) if tau_ms_off is not None else float(tau_ms))
    logger.debug("Tau ON  (ms): %s", tau_vals_on)
    logger.debug("Tau OFF (ms): %s", tau_vals_off)
    a_vals_on  = np.array([tau_to_a(tau, dt_ms) for tau in tau_vals_on])
    a_vals_off = np.array([tau_to_a(tau, dt_ms) for tau in tau_vals_off])

    # auto-set K to cover 3x the longest time constant across ON and OFF
    if K is None:
        max_tau_samples = np.max(np.concatenate([tau_vals_on, tau_vals_off])) / dt_ms
        K = int(np.ceil(3 * max_tau_samples))
        logger.debug("Auto-set K=%d samples (3 x max tau=%.1fms / dt=%sms)",
                      K, np.max(np.concatenate([tau_vals_on, tau_vals_off])), dt_ms)

    out_ON  = np.zeros((N_CFs, T))
    out_OFF = np.zeros((N_CFs, T))

    for i in range(N_CFs):
        kernel_ON  = build_ON_kernel(a_vals_on[i],  w, K)
        kernel_OFF = build_OFF_kernel(a_vals_off[i], w, K)

        # causal padding: use pad_value if given, else replicate first sample
        signal = an_output[i]
        fill   = signal[0] if pad_value is None else pad_value
        padded = np.concatenate([np.full(K - 1, fill), signal])

        raw_ON  = np.convolve(padded, kernel_ON,  mode='valid')[:T]
        raw_OFF = np.convolve(padded, kernel_OFF, mode='valid')[:T]

        onset_idx = np.argmax(np.abs(np.diff(signal)) > 0)  # first transition
        off_idx   = onset_idx + int((signal > 0).sum())
        logger.debug("CF %.0f Hz | tau_on=%.1fms | tau_off=%.1fms | K=%d",
                     CFs_Hz[i], tau_vals_on[i], tau_vals_off[i], K)
        logger.debug("  signal max:     %.4e", signal.max())
        logger.debug("  raw_ON  max:    %.4e  at t=%d", raw_ON.max(), raw_ON.argmax())
        logger.debug("  raw_ON  onset:  %.4e  (should be ~= signal.max())", raw_ON[onset_idx])
        logger.debug("  raw_OFF offset: %.4e", raw_OFF[off_idx])

        out_ON[i]  = np.convolve(padded, kernel_ON,  mode='valid')[:T]
        out_OFF[i] = np.convolve(padded, kernel_OFF, mode='valid')[:T]

    if rectify:
        out_ON  = np.maximum(out_ON,  0.0)
        out_OFF = np.maximum(out_OFF, 0.0)

    return np.stack([out_ON, out_OFF], axis=0)  # (2, N_CFs, T)


def preprocess_AN_output(an_output: np.ndarray,
                         CFs_Hz: np.ndarray,
                         dt_fine_ms: float,
                         downsample_factor: int,
                         w: float = 0.8,
                         K: Optional[int] = None) -> np.ndarray:
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


def apply_sustained_channel(
    signal: np.ndarray,
    tau_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Non-normalized causal leaky integrator (sustained neural channel).

    Implements y[n] = a·y[n-1] + x[n], a = exp(-dt/τ).

    The plateau response to amplitude A is A/(1-a) ≈ A·τ_ms/dt_ms — deliberately
    NOT normalised so the per-tone integral τ·(1-exp(-d/τ)) varies with τ.
    This τ-dependence across tone-duration conditions is what makes τ recoverable
    from BOLD amplitude differences (unlike AdapTrans whose kernel sums to 1-w
    regardless of τ).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Input boxcar train at dt_ms resolution.
    tau_ms : float
        Time constant in milliseconds.
    dt_ms : float
        Time step in milliseconds.

    Returns
    -------
    sustained : np.ndarray, shape (T,)
    """
    from scipy.signal import lfilter
    a = np.exp(-dt_ms / tau_ms)
    # Normalise by τ_samples so plateau = input amplitude (same scale as AdapTrans).
    # Without this: plateau = A/(1-a) ≈ A·τ_ms/dt_ms — up to 250× larger than AdapTrans.
    # After this: plateau → A, and per-tone integral = (1-exp(-d/τ)) — τ-sensitive but bounded.
    tau_samples = tau_ms / dt_ms
    return lfilter([1.0], [1.0, -a], signal) / tau_samples


def build_prf_boxcar_train(
    prf_responses: list,
    onsets_ms: np.ndarray,
    offsets_ms: np.ndarray,
    total_dur_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """
    Build a 1-D boxcar impulse train from per-tone pRF response scalars.

    Each tone's interval [onset, offset) in the output array is filled with
    its corresponding prf_response amplitude. All other samples are zero.

    Uses the **bare tone onset/offset times** (from result["onsets_ms"] and
    result["offsets_ms"]) — the 50 ms chunk margin does NOT affect these.

    Parameters
    ----------
    prf_responses : list of float
        One scalar per tone (mean_rate_on × duration Gaussian), length N_tones.
    onsets_ms : np.ndarray, shape (N_tones,)
        Tone onset times in milliseconds.
    offsets_ms : np.ndarray, shape (N_tones,)
        Tone offset times in milliseconds.
    total_dur_ms : float
        Total duration of the stimulus in milliseconds. Determines output length.
    dt_ms : float
        Time step in milliseconds. Default 1.0 ms.

    Returns
    -------
    train : np.ndarray, shape (ceil(total_dur_ms / dt_ms),)
        Boxcar impulse train at dt_ms resolution.
    """
    import math
    n_samples = math.ceil(total_dur_ms / dt_ms)
    train = np.zeros(n_samples)

    for s, (on, off, amp) in enumerate(zip(onsets_ms, offsets_ms, prf_responses)):
        i_on  = round(on  / dt_ms)
        i_off = round(off / dt_ms)
        # clamp to valid range
        i_on  = max(0, min(i_on,  n_samples))
        i_off = max(0, min(i_off, n_samples))
        train[i_on:i_off] = amp

    return train


def build_prf_impulse_train(
    prf_responses: list,
    onsets_ms: np.ndarray,
    total_dur_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """
    Build a 1-D impulse train from per-tone pRF response scalars.

    Each tone contributes a single delta spike at its onset sample, matching:

        x[n] = sum_s  prf_response[s] * delta[n - n_s^onset]

    so that convolving with h_ON gives:

        output[n] = sum_s  prf_response[s] * h_ON[n - n_s^onset]

    Parameters
    ----------
    prf_responses : list of float
        One scalar per tone, length N_tones.
    onsets_ms : np.ndarray, shape (N_tones,)
        Tone onset times in milliseconds.
    total_dur_ms : float
        Total duration of the stimulus in milliseconds.
    dt_ms : float
        Time step in milliseconds. Default 1.0 ms.

    Returns
    -------
    train : np.ndarray, shape (ceil(total_dur_ms / dt_ms),)
        Impulse train at dt_ms resolution.
    """
    import math
    n_samples = math.ceil(total_dur_ms / dt_ms)
    train = np.zeros(n_samples)

    for amp, on in zip(prf_responses, onsets_ms):
        i_on = round(on / dt_ms)
        i_on = max(0, min(i_on, n_samples - 1))
        train[i_on] += amp  # accumulate in case two onsets round to same sample

    return train
