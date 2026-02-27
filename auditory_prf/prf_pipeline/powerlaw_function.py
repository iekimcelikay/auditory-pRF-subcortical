import numpy as np
import sys
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.cochlea_loader_functions import load_cochlea_results, organize_for_eachtone_allCFs, resolve_results_dir
from pathlib import Path
# but i dont need a response matrix, because i'm taking each cfxsequence separately.
# ── defaults ────────────────────────────────────────────────────────────────
EXP_NAME = "dipc_test_250225_01"   # ← change this when running from the IDE

DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")

def power_function(response_matrix, alpha):
    return np.power(response_matrix, alpha)

def power_with_percf_normalization(response_matrix, alpha):
    r_powered= power_function(response_matrix, alpha)
    # Normalize each CF (each row) to its own max
    max_per_cf = np.max(r_powered, axis=1, keepdims=True)
    r_normalized = r_powered / (max_per_cf + 1e-10)
    return r_normalized

def get_cf_timecourse(data: dict, cf) -> tuple[np.ndarray, int, float]:
    """Extract a single 1-D PSTH timecourse from a loaded .npz data dict.

    Parameters
    ----------
    data : dict
        Dict returned by ``ResultSaver.load_npz``. Must contain
        ``population_rate_psth`` (num_cf, n_bins) and ``cf_list`` (num_cf,).
    cf : int or float
        * **int**   → treated as a zero-based row index into ``cf_list``.
        * **float** → treated as a CF frequency in Hz; the closest entry in
          ``cf_list`` is selected.

    Returns
    -------
    timecourse : np.ndarray, shape (n_bins,)
        Raw (untransformed) PSTH for the selected CF.
    cf_index : int
        Zero-based row index that was used.
    cf_hz : float
        Actual CF frequency in Hz for the selected row.
    """
    population_psth = data["population_rate_psth"]   # (num_cf, n_bins)
    cf_list         = np.asarray(data["cf_list"])     # (num_cf,)

    if isinstance(cf, (int, np.integer)):
        cf_index = int(cf)
        if not (0 <= cf_index < len(cf_list)):
            raise IndexError(
                f"CF index {cf_index} out of range for cf_list of length {len(cf_list)}."
            )
    else:
        cf_index = int(np.argmin(np.abs(cf_list - float(cf))))

    cf_hz      = float(cf_list[cf_index])
    timecourse = population_psth[cf_index, :]
    return timecourse, cf_index, cf_hz


def load_cf_timecourse(npz_path: Path, cf) -> tuple[np.ndarray, np.ndarray, int, float, str]:
    """Load a single CF timecourse from one .npz file (one stimulus / sequence).

    Each .npz corresponds to one stimulus (tone or sequence), and contains
    responses for all CFs.  This function loads the file and extracts the row
    for the requested CF, giving you the single 1-D timecourse you need at each
    iteration of the CF × sequence loop::

        for npz_path in sorted(results_dir.glob("*.npz")):     # ← sequence axis
            for cf in cf_values:                               # ← CF axis
                timecourse, time_axis, i_cf, cf_hz, seq_id = load_cf_timecourse(npz_path, cf)
                result = apply_powerlaw_cf(timecourse, alpha)

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file for one stimulus.
    cf : int or float
        CF selector passed through to ``get_cf_timecourse``:
        * **int**   → zero-based row index.
        * **float** → nearest CF in Hz.

    Returns
    -------
    timecourse : np.ndarray, shape (n_bins,)
        Raw PSTH for the selected CF.
    time_axis : np.ndarray, shape (n_bins,)
        Time axis in seconds.
    cf_index : int
        Zero-based CF row index that was used.
    cf_hz : float
        Actual CF frequency in Hz of the selected row.
    seq_id : str
        Stimulus identifier (``soundfileid`` key, or the file stem as fallback).
    """
    npz_path = Path(npz_path)
    saver    = ResultSaver(npz_path.parent)
    data     = saver.load_npz(npz_path.name)

    timecourse, cf_index, cf_hz = get_cf_timecourse(data, cf)
    time_axis = np.asarray(data["time_axis"])
    seq_id    = str(data.get("soundfileid", npz_path.stem))

    return timecourse, time_axis, cf_index, cf_hz, seq_id


def apply_powerlaw_cf(timecourse: np.ndarray, alpha: float) -> np.ndarray:
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
    r_powered = np.power(timecourse, alpha)
    r_normalized = r_powered / (np.max(r_powered) + 1e-10)
    return r_normalized

def apply_power_normalize(exp_name, results_dir, alpha, out_dir=None):

    # Derive paths from exp_name (CLI value overrides the module-level default)
    base_dir        = Path(f"./models_output/{exp_name}")
    out_dir_default = Path(f"./figures/{exp_name}")

    results_dir = resolve_results_dir(results_dir if results_dir is not None else base_dir)
    print(f"Experiment        : {exp_name}")
    print(f"Results directory : {results_dir}")

    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"ERROR: no .npz files found in {results_dir}")
    print(f"Found {len(npz_files)} .npz file(s)\n")

    out_dir = (out_dir if out_dir is not None else out_dir_default)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory  : {out_dir}\n")

    saver = ResultSaver(results_dir)

    for i, npz_file in enumerate(npz_files, 1):
        print(f"[{i}/{len(npz_files)}] {npz_file.name}")

        data            = saver.load_npz(npz_file.name)
        population_psth = data["population_rate_psth"]
        time_axis       = data["time_axis"]
        cf_list         = data["cf_list"]
        identifier      = str(data.get("soundfileid", npz_file.stem))


        print(f"    PSTH shape : {population_psth.shape}  |  duration : {time_axis[-1]:.2f} s")
        transformed_response = power_with_percf_normalization(population_psth, alpha)

    return transformed_response