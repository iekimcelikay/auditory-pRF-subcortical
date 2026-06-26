import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_powerlaw_population(population_psth: np.ndarray, alpha: float) -> np.ndarray:
    """Apply power-law sharpening to an array of firing rates, then rescale so
    that the grand mean is preserved (equals the pre-sharpening mean).

    Works on any ndarray shape: (n_cfs, n_bins) population PSTH or (n_tones,)
    per-tone mean rates.

    Parameters
    ----------
    population_psth : np.ndarray
        Firing rate values to sharpen.
    alpha : float
        Exponent of the power-law transformation (sharpening factor).

    Returns
    -------
    np.ndarray, same shape as input
        Sharpened rates rescaled so that
        ``np.mean(output) == np.mean(population_psth)``.
    """
    pre_mean = np.mean(population_psth)
    sharpened = np.power(population_psth, alpha)
    post_mean = np.mean(sharpened)
    logger.debug("apply_powerlaw_population: pre_mean=%.6e  post_mean=%.6e  alpha=%.2f",
                 pre_mean, post_mean, alpha)
    return sharpened * (pre_mean / (post_mean + 1e-10))
