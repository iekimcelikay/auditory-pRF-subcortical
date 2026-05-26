"""hrf.py
======
Parameterised double-gamma HRF for the subcortical auditory pRF pipeline.

Parameter names follow nipy / SPM convention (spm_hrf_compat):
    peak_delay, peak_disp, under_delay, under_disp, p_u_ratio

----------
build_hrf_kernel(...)       -> (kernel, t)
convolve_hrf(...)           -> convolved signal
spm_params / glover_params / popeye_params / subcortical_params  -> preset dicts
hrf_summary(...)            -> log kernel statistics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.stats as sps
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset parameter dicts
# ---------------------------------------------------------------------------

SPM_PARAMS = dict(
    peak_delay=6.0, peak_disp=1.0,
    under_delay=16.0, under_disp=1.0,
    p_u_ratio=6.0,
)

GLOVER_PARAMS = dict(
    peak_delay=5.4, peak_disp=0.9,
    under_delay=10.8, under_disp=0.9,
    p_u_ratio=1.0 / 0.35,
)

POPEYE_PARAMS = dict(
    peak_delay=5.4, peak_disp=0.9,
    under_delay=10.9, under_disp=0.9,
    p_u_ratio=6.0,
)

SUBCORTICAL_PARAMS = dict(
    peak_delay=5.0, peak_disp=1.0,
    under_delay=9.0, under_disp=1.0,
    p_u_ratio=6.0,
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_hrf_kernel(
    peak_delay: float  = 6.0,
    peak_disp: float   = 1.0,
    under_delay: float = 16.0,
    under_disp: float  = 1.0,
    p_u_ratio: float   = 6.0,
    dt: float          = 0.001,
    duration: float    = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalised double-gamma HRF kernel.

    h(t) = gamma(peak_delay, peak_disp)(t)
         - gamma(under_delay, under_disp)(t) / p_u_ratio

    Normalised so the positive lobe has area = 1.

    Parameters
    ----------
    peak_delay : float
        Delay of the peak gamma (seconds). SPM default: 6.
    peak_disp : float
        Dispersion (scale) of the peak gamma. SPM default: 1.
    under_delay : float
        Delay of the undershoot gamma (seconds). SPM default: 16.
    under_disp : float
        Dispersion of the undershoot gamma. SPM default: 1.
    p_u_ratio : float
        Peak-to-undershoot ratio. Larger = smaller undershoot. SPM default: 6.
    dt : float
        Time resolution in seconds (default 0.001).
    duration : float
        Kernel duration in seconds (default 32).

    Returns
    -------
    kernel : np.ndarray, shape (n_samples,)
        Normalised HRF kernel.
    t : np.ndarray, shape (n_samples,)
        Time axis in seconds.
    """
    t = np.arange(0.0, duration, dt)

    peak  = sps.gamma.pdf(t, peak_delay / peak_disp, loc=0, scale=peak_disp)
    under = sps.gamma.pdf(t, under_delay / under_disp, loc=0, scale=under_disp)

    kernel   = peak - under / p_u_ratio
    pos_area = np.sum(kernel[kernel > 0]) * dt

    if pos_area > 0:
        kernel = kernel / pos_area

    return kernel, t


def convolve_hrf(
    signal: np.ndarray,
    kernel: np.ndarray,
    signal_dt: float,
    kernel_dt: float   = 0.001,
    duration: float    = 32.0,
    output_dt: Optional[float] = None,
) -> np.ndarray:
    """Convolve a neural signal with an HRF kernel and optionally downsample.

    If the signal and kernel have different time resolutions, the kernel is
    resampled to match ``signal_dt`` via linear interpolation.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D neural signal.
    kernel : np.ndarray, shape (m,)
        HRF kernel (e.g. from ``build_hrf_kernel``).
    signal_dt : float
        Resolution of the input signal in seconds.
    kernel_dt : float
        Resolution of the kernel in seconds (default 0.001).
    duration : float
        Kernel duration in seconds (default 32). Used only when resampling.
    output_dt : float, optional
        Downsample target (e.g. TR). ``None`` keeps *signal_dt*.

    Returns
    -------
    conv : np.ndarray
        Convolved (and possibly downsampled) signal.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-D, got shape {signal.shape}."
        )

    # Resample kernel if resolutions differ
    if not np.isclose(signal_dt, kernel_dt):
        t_orig = np.arange(0.0, duration, kernel_dt)
        t_new  = np.arange(0.0, duration, signal_dt)
        kernel = np.interp(t_new, t_orig, kernel)

    conv = fftconvolve(signal, kernel, mode="full")[:len(signal)] * signal_dt

    if output_dt is not None and not np.isclose(output_dt, signal_dt):
        n    = int(round(output_dt / signal_dt))
        conv = conv[::n]

    return conv


def hrf_summary(
    kernel: np.ndarray,
    t: np.ndarray,
    dt: float,
    params: dict,
) -> None:
    """Log kernel statistics (peak time, FWHM, undershoot, area).

    Parameters
    ----------
    kernel : np.ndarray, shape (n_samples,)
        HRF kernel.
    t : np.ndarray, shape (n_samples,)
        Time axis in seconds.
    dt : float
        Time resolution used to build the kernel.
    params : dict
        Parameter dict (must contain ``peak_delay`` and ``peak_disp``).
    """
    peak1  = params["peak_delay"] - params["peak_disp"]
    peak_t = t[np.argmax(kernel)]
    above  = t[kernel >= np.max(kernel) / 2.0]
    fwhm   = float(above[-1] - above[0]) if len(above) > 1 else float("nan")

    logger.info("  peak_delay=%.1f  peak_disp=%.1f  under_delay=%.1f  "
                "under_disp=%.1f  p_u_ratio=%.2f",
                params["peak_delay"], params["peak_disp"],
                params["under_delay"], params["under_disp"],
                params["p_u_ratio"])
    logger.info("  Kernel peak:    %.3f s  (analytic peak1: %.3f s)",
                peak_t, peak1)
    logger.info("  Kernel FWHM:    %.3f s", fwhm)
    logger.info("  Undershoot min: %.4f", np.min(kernel))
    logger.info("  Positive area:  %.4f  (should be 1.0)",
                np.sum(kernel[kernel > 0]) * dt)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from auditory_prf.utils.logging_configurator import LoggingConfigurator

    DEFAULT_OUT_DIR = Path("./figures/hrf")

    p = argparse.ArgumentParser(description="HRF smoke test")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dpi",     type=int,  default=150)
    args = p.parse_args()

    out_dir = args.out_dir.expanduser().resolve()

    LoggingConfigurator(
        output_dir=out_dir,
        log_filename="hrf_smoketest.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    ).setup()

    presets = {
        "SPM":         SPM_PARAMS,
        "Glover":      GLOVER_PARAMS,
        "Popeye":      POPEYE_PARAMS,
        "Subcortical": SUBCORTICAL_PARAMS,
    }

    dt       = 0.001
    duration = 32.0

    logger.info("=== HRF summaries ===")
    kernels = {}
    for name, params in presets.items():
        kernel, t = build_hrf_kernel(**params, dt=dt, duration=duration)
        kernels[name] = (kernel, t)
        logger.info("--- %s ---", name)
        hrf_summary(kernel, t, dt, params)

    # Convolution demo
    TR, neural_dt = 1.0, 0.001
    t_n = np.arange(0, 20, neural_dt)
    sig = np.zeros_like(t_n)
    sig[(t_n >= 2) & (t_n < 4)]   = 1.0
    sig[(t_n >= 12) & (t_n < 14)] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for name, (kernel, t) in kernels.items():
        ax.plot(t, kernel, label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Canonical HRFs")
    ax.legend()

    ax = axes[1]
    spm_kernel, _ = build_hrf_kernel(**SPM_PARAMS, dt=neural_dt)
    bold = convolve_hrf(sig, spm_kernel, signal_dt=neural_dt,
                        kernel_dt=neural_dt, output_dt=TR)
    t_b = np.arange(len(bold)) * TR
    ax.fill_between(t_n, sig * 0.15, alpha=0.3, label="Neural (scaled)")
    ax.plot(t_b, bold, "o-", ms=3, label="BOLD (SPM)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_title("Convolution test")

    fig.tight_layout()
    out_path = out_dir / "hrf_test.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out_path)
