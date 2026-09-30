"""Shared helpers for reproducing Lerma-Usabiaga et al. Fig 4's SNR paradigm.

Both tests/test_pm_noise.py (Test 48) and tests/plot_noise_verification.py
(Fig 3) check whether pm_noise.py's REFERENCE_BOLD_STD reproduces the
paper's published SNRs, using the paper's own Methods description:

    TR = 1.5 s, 24 s cycle (0.042 Hz stimulus frequency), 8 cycles/run,
    calibration voxels selected for 8-12% peak-to-peak PSC contrast,
    single acquisition per run. "SNR was calculated by the ratio of the
    root mean squared error of the signal (the fitted sinusoidal) and the
    noise (time series minus the fitted sinusoidal)."

This module is the single source of truth for that paradigm so the two
call sites can't drift apart.
"""
import numpy as np

from prf_models.pm_noise import PmAdapter

TR_S = 1.5
CYCLE_PERIOD_S = 24.0
N_CYCLES = 8
STIMULUS_FREQ_HZ = 1.0 / CYCLE_PERIOD_S
PSC_AMPLITUDE = 0.05  # peak amplitude -> 10% peak-to-peak, mid of the paper's 8-12% PSC range
N_TR = round(N_CYCLES * CYCLE_PERIOD_S / TR_S)

FIG4_TARGETS_DB = {'low': 5.29, 'mid': -0.51, 'high': -4.29}


def build_clean_bold_and_pm() -> tuple[np.ndarray, PmAdapter]:
    """Build the paper's calibration stimulus: an 8-cycle PSC sinusoid.

    Returns
    -------
    clean_bold_psc : np.ndarray, shape (N_TR,)
        Fractional PSC sinusoid at STIMULUS_FREQ_HZ, amplitude PSC_AMPLITUDE.
    pm : PmAdapter
        Matching pm_noise.py adapter (TR_S, N_TR TRs).
    """
    t_s = np.arange(N_TR) * TR_S
    clean_bold_psc = PSC_AMPLITUDE * np.sin(2 * np.pi * STIMULUS_FREQ_HZ * t_s)
    pm = PmAdapter(TR=TR_S, time_points_n=N_TR, time_points_series=t_s)
    return clean_bold_psc, pm


def fit_sinusoid_snr_db(noisy_signal_psc: np.ndarray) -> float:
    """SNR (dB) via harmonic-fit residual, matching the paper's Methods.

    Fits a single sinusoid at STIMULUS_FREQ_HZ (amplitude, phase, and DC
    offset free) by least squares, then computes
    ``20*log10(rms(fit) / rms(noisy_signal_psc - fit))``.

    Parameters
    ----------
    noisy_signal_psc : np.ndarray, shape (N_TR,)
        BOLD signal in fractional PSC units (0.05 == 5%).

    Returns
    -------
    float
        SNR in dB.
    """
    n_tr = len(noisy_signal_psc)
    t_s = np.arange(n_tr) * TR_S
    basis = np.column_stack([
        np.sin(2 * np.pi * STIMULUS_FREQ_HZ * t_s),
        np.cos(2 * np.pi * STIMULUS_FREQ_HZ * t_s),
        np.ones(n_tr),
    ])
    coeffs, *_ = np.linalg.lstsq(basis, noisy_signal_psc, rcond=None)
    fit = basis @ coeffs
    residual = noisy_signal_psc - fit
    return 20.0 * np.log10(np.sqrt(np.mean(fit ** 2)) / np.sqrt(np.mean(residual ** 2)))
