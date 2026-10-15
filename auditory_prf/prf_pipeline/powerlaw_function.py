from typing import Iterable, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_population_scale(population_psths: Iterable[np.ndarray], alpha: float) -> float:
    """Rescale factor ``mean(pool) / mean(pool ** alpha)`` for a pre-power-law
    population pool spanning many sequences (silence included).

    The pool is streamed (running sums in float64), so sequences never need to
    be held in memory together. Recompute for every alpha: the post-sharpening
    side depends on alpha.

    Parameters
    ----------
    population_psths : iterable of np.ndarray
        One raw (unsharpened) array per sequence, e.g. (n_cf, n_bins) population
        PSTHs from every .npz file, including the silence sequence.
    alpha : float
        Exponent of the power-law transformation.

    Returns
    -------
    float
        Multiplier applied after ``x ** alpha``; 1.0 if the pool is empty or
        ``sum(pool ** alpha)`` is not positive.
    """
    pre_sum, post_sum, n_values = 0.0, 0.0, 0
    for psth in population_psths:
        psth = np.asarray(psth, dtype=np.float64)
        pre_sum += float(psth.sum())
        post_sum += float(np.power(psth, alpha).sum())
        n_values += psth.size
    if n_values == 0 or post_sum <= 0:
        return 1.0
    scale = pre_sum / post_sum
    logger.debug("compute_population_scale: n=%d  pre_mean=%.6e  post_mean=%.6e  alpha=%.2f  scale=%.6e",
                 n_values, pre_sum / n_values, post_sum / n_values, alpha, scale)
    return scale


def apply_powerlaw_population(
    population_psth: np.ndarray,
    alpha: float,
    population_scale: Optional[float] = None,
) -> np.ndarray:
    """Apply power-law sharpening to an array of firing rates, then rescale.

    Works on any ndarray shape: (n_cfs, n_bins) population PSTH or (n_tones,)
    per-tone mean rates.

    Parameters
    ----------
    population_psth : np.ndarray
        Firing rate values to sharpen.
    alpha : float
        Exponent of the power-law transformation (sharpening factor).
    population_scale : float, optional
        Rescale factor from ``compute_population_scale`` (pre-power-law
        population pool over all sequences incl. silence). If None, falls back
        to self-normalizing on ``population_psth`` alone.

    Returns
    -------
    np.ndarray, same shape as input
        ``population_psth ** alpha * population_scale`` if a scale is given;
        otherwise rescaled so ``np.mean(output) == np.mean(population_psth)``.
    """
    if population_scale is not None:
        return np.power(population_psth, alpha) * population_scale
    pre_mean = np.mean(population_psth)
    sharpened = np.power(population_psth, alpha)
    post_mean = np.mean(sharpened)
    logger.debug("apply_powerlaw_population: pre_mean=%.6e  post_mean=%.6e  alpha=%.2f",
                 pre_mean, post_mean, alpha)
    return sharpened * (pre_mean / (post_mean + 1e-10))
