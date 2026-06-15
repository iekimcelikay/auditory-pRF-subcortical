from typing import Optional

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
                    tau_ms: Optional[float] = None,
                    cf_range_hz: Optional[tuple] = None,
                    tau_range_ms: tuple = (10.0, 500.0)) -> np.ndarray:
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
        Half-wave rectify output (ReLU). Default False. #TODO: make it false.
    # NOTE: however we actually need to rectify the output, why do i make the default false?

    pad_value : float or None
        Value used to pad the left edge of each channel before convolution.
        If None (default), replicates signal[0] of each channel (standard
        causal padding). Pass 0.0 for isolated per-tone signals that start
        with non-zero amplitude at t=0 to avoid suppressing the first onset.
    tau_ms : float or None
        If given, use this fixed time constant (ms) for every CF, bypassing
        ``cf_to_tau_ms``. Useful for quick tests (e.g. ``tau_ms=50``).
    cf_range_hz : tuple of (min_hz, max_hz) or None
        CF range spanned by the experiment, used by ``cf_to_tau_ms`` to map
        each CF to a tau. If None, defaults to ``(CFs_Hz.min(), CFs_Hz.max())``.
        Ignored if ``tau_ms`` is given.
    tau_range_ms : tuple of (tau_min_ms, tau_max_ms)
        Time-constant range passed to ``cf_to_tau_ms``. Default (10.0, 500.0):
        the lowest CF gets 500ms, the highest CF gets 10ms.
        Ignored if ``tau_ms`` is given.

    Returns
    -------
    out : np.ndarray, shape (2, N_CFs, T)
        out[0] = ON  (onset)  responses
        out[1] = OFF (offset) responses
    """
    N_CFs, T = an_output.shape

    # per-CF time constants and decay rates
    if tau_ms is not None:
        tau_vals = np.full(N_CFs, float(tau_ms))
    else:
        if cf_range_hz is None:
            cf_range_hz = (float(np.min(CFs_Hz)), float(np.max(CFs_Hz)))
        tau_vals = np.array([cf_to_tau_ms(cf, cf_range_hz, tau_range_ms) for cf in CFs_Hz])
    print(f"Tau for this CF is: {tau_vals}")
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

        # causal padding: use pad_value if given, else replicate first sample
        signal = an_output[i]
        fill   = signal[0] if pad_value is None else pad_value
        padded = np.concatenate([np.full(K - 1, fill), signal])

        raw_ON  = np.convolve(padded, kernel_ON,  mode='valid')[:T]
        raw_OFF = np.convolve(padded, kernel_OFF, mode='valid')[:T]

        # ── ADD THIS ─────────────────────────────────────────────────
        onset_idx = np.argmax(np.abs(np.diff(signal)) > 0)  # first transition
        print(f"CF {CFs_Hz[i]:.0f} Hz | tau={tau_vals[i]:.1f}ms | K={K}")
        print(f"  signal max:     {signal.max():.4e}")
        print(f"  raw_ON  max:    {raw_ON.max():.4e}  at t={raw_ON.argmax()}")
        print(f"  raw_ON  onset:  {raw_ON[onset_idx]:.4e}  (should be ≈ signal.max())")
        off_idx = onset_idx + int((signal > 0).sum())
        print(f"  raw_OFF offset: {raw_OFF[off_idx]:.4e}")
        # ─────────────────────────────────────────────────────────────


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
