import numpy as np
import sys
import logging
from pathlib import Path

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.cochlea_loader_functions import load_cochlea_results, organize_for_eachtone_allCFs, resolve_results_dir

logger = logging.getLogger(__name__)

def power_function(response_matrix, alpha):
    return np.power(response_matrix, alpha)

def power_with_percf_normalization(response_matrix, alpha):
    # NOTE: OLD FUNCTION!! DEPRECATED!!!

    r_powered= power_function(response_matrix, alpha)
    # Normalize each CF (each row) to its own max
    max_per_cf = np.max(r_powered, axis=1, keepdims=True)
    r_normalized = r_powered / (max_per_cf + 1e-10)
    return r_normalized

def apply_powerlaw_population(population_psth: np.ndarray, alpha: float) -> np.ndarray:
    # NOTE: this is the function that is called in the pipeline.
    """Apply power-law sharpening to the full population PSTH, then rescale so
    that the grand mean across all cochlear channels and all time bins is
    preserved (equals the pre-sharpening mean).

    Parameters
    ----------
    population_psth : np.ndarray, shape (n_cfs, n_bins)
        Raw PSTH matrix for all cochlear channels (CFs) and all time bins for
        one stimulus sequence (one .npz file).
    alpha : float
        Exponent of the power-law transformation (sharpening factor).

    Returns
    -------
    np.ndarray, shape (n_cfs, n_bins)
        Sharpened population PSTH rescaled so that
        ``np.mean(output) == np.mean(population_psth)``.
    """
    if population_psth.ndim != 2:
        raise ValueError(
            f"population_psth must be 2-D (n_cfs, n_bins), got shape {population_psth.shape}."
        )
    pre_mean = np.mean(population_psth)
    sharpened = np.power(population_psth, alpha)
    post_mean = np.mean(sharpened)
    logger.debug("apply_powerlaw_population: pre_mean=%.6e  post_mean=%.6e  alpha=%.2f",
                 pre_mean, post_mean, alpha)
    return sharpened * (pre_mean / (post_mean + 1e-10))


def apply_powerlaw_cf(timecourse: np.ndarray, alpha: float) -> np.ndarray:
    # NOTE: OLD FUNCTION!! DEPRECATED!!!

    """Apply power-law compression and max-normalization to a single CF timecourse.

    Parameters
    ----------
    timecourse : np.ndarray, shape (n_bins,)
        PSTH timecourse for a single CF from one .npz file
        (i.e. one row of ``population_rate_psth[i_cf, :]``).
    alpha : float
        Exponent of the power-law transformation.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Power-law compressed and max-normalised timecourse with values in [0, 1].
    """
    if timecourse.ndim != 1:
        raise ValueError(
            f"timecourse must be 1-D, got shape {timecourse.shape}. "
            "Pass a single row: population_rate_psth[i_cf, :]"
        )
    logger.debug("max_firing rate of timecourse: %s", np.max(timecourse))
    r_powered = np.power(timecourse, alpha)
    logger.debug("max_firing rate of r_powered: %s", np.max(r_powered))
    r_normalized = r_powered / (np.max(r_powered) + 1e-10)
    return r_normalized
    #return r_powered



## OLD FUNCTION (CURRENTLY NOT USED IN THE PIPELINE):
# Module-level logger -- no handlers configured here.
# relies on LoggingConfigurator called in the pipeline entry point.

# ── defaults ────────────────────────────────────────────────────────────────
EXP_NAME = "dipc_test_250225_01"   # ← change this when running from the IDE

DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")
def apply_power_normalize(exp_name, results_dir, alpha, out_dir=None):

    # Derive paths from exp_name (CLI value overrides the module-level default)
    base_dir        = Path(f"./models_output/{exp_name}")
    out_dir_default = Path(f"./figures/{exp_name}")

    results_dir = resolve_results_dir(results_dir if results_dir is not None else base_dir)
    logger.info("Experiment        :%s", exp_name)
    logger.info("Results directory : %s", results_dir)

    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        logger.error("No .npz files found in %s", results_dir)
        sys.exit(1)
    logger.info("Found %d .npz file(s)", len(npz_files))

    out_dir = (out_dir if out_dir is not None else out_dir_default)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory  : %s", out_dir)

    saver = ResultSaver(results_dir)

    for i, npz_file in enumerate(npz_files, 1):
        logger.info("[%d/%d] %s", i, len(npz_files), npz_file.name)


        data            = saver.load_npz(npz_file.name)
        population_psth = data["population_rate_psth"]
        time_axis       = data["time_axis"]
        cf_list         = data["cf_list"]
        identifier      = str(data.get("soundfileid", npz_file.stem))


        logger.debug("  PSTH shape: %s | duration: %.2f s",
                     population_psth.shape, time_axis[-1])
        transformed_response = power_with_percf_normalization(population_psth, alpha)

    return transformed_response