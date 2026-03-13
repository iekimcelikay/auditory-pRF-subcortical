import numpy as np
import sys
from pathlib import Path

from auditory_prf.utils.result_saver import ResultSaver

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


def load_population_psth(npz_path: Path, cf) -> tuple[np.ndarray, np.ndarray, int, float, str]:
    """Load the full population PSTH matrix from one .npz file, plus CF metadata.

    Use this upstream of ``apply_powerlaw_population`` so that the
    mean-preserving sharpening normalization is computed over all cochlear
    channels (all CFs × all time bins) for a single stimulus sequence.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file for one stimulus sequence.
    cf : int or float
        CF selector passed through to ``get_cf_timecourse``:
        * **int**   → zero-based row index.
        * **float** → nearest CF in Hz.

    Returns
    -------
    population_psth : np.ndarray, shape (n_cfs, n_bins)
        Full raw PSTH matrix for all cochlear channels.
    time_axis : np.ndarray, shape (n_bins,)
        Time axis in seconds.
    cf_index : int
        Zero-based CF row index resolved from ``cf``.
    cf_hz : float
        Actual CF frequency in Hz for the selected row.
    seq_id : str
        Stimulus identifier (``soundfileid`` key, or the file stem as fallback).
    """
    npz_path = Path(npz_path)
    saver    = ResultSaver(npz_path.parent)
    data     = saver.load_npz(npz_path.name)

    _, cf_index, cf_hz = get_cf_timecourse(data, cf)
    population_psth = np.asarray(data["population_rate_psth"])   # (n_cfs, n_bins)
    time_axis       = np.asarray(data["time_axis"])
    seq_id          = str(data.get("soundfileid", npz_path.stem))

    return population_psth, time_axis, cf_index, cf_hz, seq_id
