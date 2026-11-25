"""highpass.py
===========
SPM-style high-pass filter (discrete cosine transform) of the recovery scripts.

The filter removes the slow DCT components (period longer than a cutoff) of a time course. It is applied to the
noisy measurement and to the templates of every run, with the same projection, as SPM applies its filter to the
data and to the design matrix.

Constant column: SPM drops the constant from its filter (``X0 = X0(:,2:end)``) and fits it as a separate GLM
regressor. ``highpass_projection`` removes the constant too. Under the Pearson-r matching of the recovery scripts
(mean-removed measurement and templates) the scores are identical either way (checked numerically on toy data).
The SPM-exact version (constant column dropped, the constant fitted as an explicit nuisance column) is only needed
if the metric changes to one that does not remove the mean.

Run from the project root.
"""

import math
from typing import Optional, Sequence

import numpy as np

DEFAULT_HIGHPASS_CUTOFFS_S = [128.0]      # the SPM default cutoff, in seconds


def n_highpass_columns(n_tr: int, tr_s: float, cutoff_s: float) -> int:
    """Number of DCT drift columns (the constant included) with a period longer than ``cutoff_s``.

    SPM convention: ``floor(2 * n_tr * tr_s / cutoff_s) + 1``.
    """
    return int(np.floor(2.0 * n_tr * tr_s / cutoff_s)) + 1


def highpass_projection(n_tr: int, tr_s: float, cutoff_s: float) -> np.ndarray:
    """Matrix P that removes the slow DCT components of a time course of length ``n_tr``.

    ``P = I - B B^T`` with B the orthonormal DCT-II columns 0 .. K-1 (K = ``n_highpass_columns``); column 0
    is the constant, so P also removes the mean. P is symmetric and idempotent.

    Returns
    -------
    np.ndarray float64, shape (n_tr, n_tr)
    """
    n_columns = n_highpass_columns(n_tr, tr_s, cutoff_s)
    i = np.arange(n_tr)[:, None]
    k = np.arange(n_columns)[None, :]
    basis = np.sqrt(2.0 / n_tr) * np.cos(np.pi * (2 * i + 1) * k / (2 * n_tr))
    basis[:, 0] = 1.0 / np.sqrt(n_tr)
    return np.eye(n_tr) - basis @ basis.T


def highpass_template_per_run(template_per_run: list, projection: np.ndarray) -> list:
    """High-pass every run's templates with the same projection; the amplitudes are kept.

    Parameters
    ----------
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)
    projection : np.ndarray, shape (n_tr, n_tr)
        ``highpass_projection``; it also removes the mean.

    Returns
    -------
    list of np.ndarray float32, each (n_candidates, n_tr); pass it to ``normalize_run_templates`` like the
    unfiltered templates.
    """
    return [(templates.astype(np.float64) @ projection).astype(np.float32) for templates in template_per_run]


def filter_name(cutoff_s: Optional[float]) -> str:
    """Name of a filter in file and folder names: ``None`` -> ``'none'``, ``128.0`` -> ``'hp128'``."""
    return "none" if cutoff_s is None else f"hp{cutoff_s:g}"


def parse_cutoffs(cutoffs_s: Sequence[float]) -> list:
    """Cutoffs (seconds) from the command line: duplicates dropped (first order kept), 0 = no filter, dropped.

    Raises
    ------
    ValueError
        For a negative or non-finite (nan, inf) cutoff.
    """
    if any(not math.isfinite(cutoff) or cutoff < 0 for cutoff in cutoffs_s):
        raise ValueError(f"high-pass cutoffs must be finite and >= 0 seconds, got {list(cutoffs_s)}")
    return [cutoff for cutoff in dict.fromkeys(cutoffs_s) if cutoff > 0]


def check_cutoff(n_tr: int, tr_s: float, cutoff_s: float) -> None:
    """Raise ``ValueError`` unless the cutoff makes a real filter for a run of ``n_tr`` volumes.

    The cutoff must be > 0 and leave at most ``n_tr`` DCT columns (``n_highpass_columns``). With a cutoff of
    2 * ``tr_s`` or less there are more columns than samples, the columns wrap around, ``B`` is no longer
    orthonormal and ``P = I - B B^T`` is no projection (typical mistake: a cutoff typed in Hz instead of seconds).
    """
    if not cutoff_s > 0:                      # also catches nan
        raise ValueError(f"the high-pass cutoff must be > 0 seconds, got {cutoff_s}")
    n_columns = n_highpass_columns(n_tr, tr_s, cutoff_s)
    if n_columns > n_tr:
        raise ValueError(f"a {cutoff_s:g} s cutoff needs {n_columns} DCT columns for {n_tr} volumes of {tr_s:g} s: "
                         f"the cutoff must be longer than 2 x TR = {2 * tr_s:g} s (seconds, not Hz)")
