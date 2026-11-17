"""additive_noise.py
=================
Constants and helpers for the MATLAB-style ADDITIVE noise of the sweep recovery scripts.

Original (vistalab/PRFmodel, prfModel.m ``compute``): ``BOLDnoise = BOLDmeanValue * (1 + 2 * BOLDcontrast/100
* BOLDconv + Noise.values)``. The noise is added as is, in fractions of the mean, so it does not depend on how
strong the voxel's response is; the signal's size comes from a fixed best case (full stimulation = 1) times the
user-set contrast (default 5 -> 10 % peak to peak). Here:

    measurement = SIGNAL_SCALE * bold + PmNoise.values,      SIGNAL_SCALE = BEST_CASE_FRACTION / BEST_CASE_PEAK_TO_PEAK

Provenance of ``BEST_CASE_PEAK_TO_PEAK`` (full derivation and checks: auditory-prf-docs-all/additive_noise_design.md)
----------------------------------------------------------------------------------------------------------------
* Grid: ``run_grid_design1000.npz``, key ``template_matrix_mean_removed``: 7,560 clean BOLD time courses
  (30 CF x 7 alpha x 9 tau x 4 w; 408 TRs of 1.6 s) in the forward model's arbitrary units.
* Peak-to-peak of each candidate (``max - min``, as MATLAB defines the contrast); the maximum over candidates is
  1899.8776, the candidate (cf_index 3, alpha 8, tau 20 ms, w 0.7): the strongest response the parameter ranges can
  give to this stimulus set = the "best case".
* It is the same for every run order within 2 % (designs 1000-1023: 1895-1940), so one constant is used for all.
* Cross-check: after scaling, that candidate has std 0.03225; the Python port's REFERENCE_BOLD_STD (0.0323) was
  fitted independently so that the low/mid/high presets reproduce the SNRs of Lerma-Usabiaga et al. Fig 4.
* Recompute it (``best_case_peak_to_peak``) when the stimulus set, the run design, the parameter ranges (alpha > 8,
  other CFs / w / tau) or the forward model change.
"""

from typing import Optional

import numpy as np

BOLD_CONTRAST_PERCENT = 5.0                                  # MATLAB default BOLDcontrast
BEST_CASE_FRACTION = 2 * BOLD_CONTRAST_PERCENT / 100         # 0.10 = 10 % peak to peak for the best case
BEST_CASE_PEAK_TO_PEAK = 1899.8776                           # model units, see the module docstring
SIGNAL_SCALE = BEST_CASE_FRACTION / BEST_CASE_PEAK_TO_PEAK   # fraction of the mean per model unit
NOISE_MODELS = ("additive", "legacy")


def best_case_peak_to_peak(template_matrix: np.ndarray) -> float:
    """Largest peak-to-peak of any candidate (the 'best case') in model units.

    Parameters
    ----------
    template_matrix : np.ndarray, shape (n_candidates, n_tr)
        Clean BOLD of every candidate (mean-removed or not; the range does not depend on the mean).

    Returns
    -------
    float
    """
    assert template_matrix.ndim == 2, template_matrix.shape
    return float((template_matrix.max(axis=1) - template_matrix.min(axis=1)).max())


def signal_scale_for(noise_model: str) -> Optional[float]:
    """Signal scale of a noise model: SIGNAL_SCALE for 'additive', None (legacy per-voxel scaling) for 'legacy'."""
    if noise_model == "additive":
        return SIGNAL_SCALE
    if noise_model == "legacy":
        return None
    raise ValueError(f"noise model must be one of {NOISE_MODELS}, got {noise_model!r}")
