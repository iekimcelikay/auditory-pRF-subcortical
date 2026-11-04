"""run_assembly_sweeps.py
========================
Run-level forward model for continuous FM-sweep stimuli: place each trial's sharpened
cochlear response in the run, run AdapTrans once over the whole run, convolve with the HRF.

Mirrors run_assembly.assemble_run_bold, with three differences:
  - trials are continuous 1 ms rate traces of one CF (20000 bins, silence tail included),
    not boxcar trains; null trials place the silent trial's response (SILENCE_SEQ_ID)
  - AdapTrans pads the left edge of the run with the FIRST SAMPLE (the run opens at the
    spontaneous rate, so zero padding would create a fake onset transient at t = 0)
  - AdapTrans is apply_adaptrans_continuous (full_pipeline_sweeps_adaptrans), not the shared
    apply_adaptrans, whose boxcar-specific debug index fails on run-level traces that are
    positive in every sample; run_assembly.py and adaptrans_onoff_filters.py stay untouched

Stages for one voxel (cf_index, alpha, tau_ms, w, rho):
  raw PSTHs (all stimuli + silence)
    -> power-law sharpening, one pool scale over all stimuli including silence
    -> CF row per stimulus -> run-level train (blanks at the silence mean rate)
    -> AdapTrans ON/OFF -> HRF -> TR sampling -> bold_combined = rho * ON + OFF
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from auditory_prf.prf_pipeline.full_pipeline_sweeps_adaptrans import (
    ADAPTRANS_K,
    ADAPTRANS_RECTIFY,
    ADAPTRANS_TAU_MS,
    ADAPTRANS_TAU_MS_OFF,
    ADAPTRANS_W,
    ALPHA,
    apply_adaptrans_continuous,
    load_sweep_result,
)
from auditory_prf.prf_pipeline.hrf import SUBCORTICAL_PARAMS, build_hrf_kernel, convolve_hrf
from auditory_prf.prf_pipeline.powerlaw_function import (
    apply_powerlaw_population,
    compute_population_scale,
)

logger = logging.getLogger(__name__)

SIGNAL_DT_S = 1e-3
BOLD_RHO = 1.0


# ── Loading ───────────────────────────────────────────────────────────────────
def load_raw_psths(results_dir: Path) -> tuple:
    """Load every combined .npz of a results folder, keyed by soundfileid.

    Returns
    -------
    raw_psth_by_seq : dict {soundfileid: (n_cf, n_bins) spikes/s}
    cf_list_hz : np.ndarray, shape (n_cf,)
    """
    raw_psth_by_seq, cf_list_hz = {}, None
    for path in sorted(Path(results_dir).glob("*.npz")):
        result = load_sweep_result(path)
        if cf_list_hz is None:
            cf_list_hz = result["cf_list"]
        assert np.array_equal(result["cf_list"], cf_list_hz), f"CF list differs in {path.name}"
        raw_psth_by_seq[result["soundfileid"]] = result["population_rate_psth"]
    assert raw_psth_by_seq, f"no .npz files in {results_dir}"
    return raw_psth_by_seq, cf_list_hz


# ── Preprocessing ─────────────────────────────────────────────────────────────
def sharpen_cf_rows(raw_psth_by_seq: dict, cf_index: int, alpha: float,
                    population_scale: Optional[float] = None) -> dict:
    """Power-law sharpen every stimulus, return the selected CF's trace of each.

    The pool scale is mean(pool) / mean(pool ** alpha) over ALL CFs of ALL stimuli
    (silence included); sharpening is elementwise, so taking the CF row afterwards is
    identical to sharpening the whole array first. Pass population_scale to reuse a
    scale already computed for this alpha (it is the same for every CF).

    Returns
    -------
    dict {soundfileid: (n_bins,)}
    """
    if population_scale is None:
        population_scale = compute_population_scale(raw_psth_by_seq.values(), alpha)
    return {seq_id: apply_powerlaw_population(psth[cf_index], alpha, population_scale)
            for seq_id, psth in raw_psth_by_seq.items()}


# ── Analysis ──────────────────────────────────────────────────────────────────
def assemble_sweep_run_train(
    per_seq_train: dict,
    run_design: list,
    total_run_dur_s: float,
    spont_rate: float,
    signal_dt_s: float = SIGNAL_DT_S,
) -> np.ndarray:
    """Place each trial's 1-D trace at its onset; fill the run's blanks with spont_rate.

    Parameters
    ----------
    per_seq_train : dict {soundfileid: (n_bins,)}
    run_design : list of (soundfileid, onset_s); nulls use the silence id (no None).
    spont_rate : float
        Rate used before the first and after the last trial (mean silence response).

    Returns
    -------
    np.ndarray, shape (round(total_run_dur_s / signal_dt_s),)
    """
    n_samples = int(round(total_run_dur_s / signal_dt_s))
    full_train = np.zeros(n_samples)
    first_onset, last_end = n_samples, 0
    for seq_id, onset_s in run_design:
        assert seq_id is not None and seq_id in per_seq_train, f"no response for '{seq_id}'"
        onset = int(round(onset_s / signal_dt_s))
        train = per_seq_train[seq_id]
        end = onset + len(train)
        assert end <= n_samples, f"'{seq_id}' extends past the run end"
        full_train[onset:end] += train
        first_onset, last_end = min(first_onset, onset), max(last_end, end)
    full_train[:first_onset] = spont_rate
    full_train[last_end:] = spont_rate
    return full_train


def sweep_run_bold(
    full_train: np.ndarray,
    hrf_kernel: np.ndarray,
    tr_s: float,
    w: float = ADAPTRANS_W,
    K: Optional[int] = ADAPTRANS_K,
    rectify: bool = ADAPTRANS_RECTIFY,
    rho: float = BOLD_RHO,
    tau_ms: float = ADAPTRANS_TAU_MS,
    tau_ms_off: Optional[float] = ADAPTRANS_TAU_MS_OFF,
    signal_dt_s: float = SIGNAL_DT_S,
) -> dict:
    """AdapTrans ON/OFF over the whole run (first-sample padding), HRF, TR sampling.

    Returns
    -------
    dict with on_response, off_response (n_samples,), bold_on, bold_off, bold_combined
    (n_tr,) and t_tr (n_tr,) seconds. bold_combined = rho * bold_on + bold_off.
    """
    assert full_train.ndim == 1
    on, off, _ = apply_adaptrans_continuous(full_train[np.newaxis, :], signal_dt_s * 1000.0,
                                            w, K, rectify, tau_ms, tau_ms_off)
    bold_on = convolve_hrf(on[0], hrf_kernel, signal_dt=signal_dt_s, kernel_dt=signal_dt_s,
                           output_dt=tr_s)
    bold_off = convolve_hrf(off[0], hrf_kernel, signal_dt=signal_dt_s, kernel_dt=signal_dt_s,
                            output_dt=tr_s)
    return {"on_response": on[0], "off_response": off[0],
            "bold_on": bold_on, "bold_off": bold_off,
            "bold_combined": rho * bold_on + bold_off,
            "t_tr": np.arange(len(bold_on)) * tr_s}


def run_sweep_forward_model(
    raw_psth_by_seq: dict,
    cf_index: int,
    run_design: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    alpha: float = ALPHA,
    w: float = ADAPTRANS_W,
    K: Optional[int] = ADAPTRANS_K,
    rectify: bool = ADAPTRANS_RECTIFY,
    rho: float = BOLD_RHO,
    tau_ms: float = ADAPTRANS_TAU_MS,
    tau_ms_off: Optional[float] = ADAPTRANS_TAU_MS_OFF,
    hrf_kernel: Optional[np.ndarray] = None,
) -> dict:
    """Clean BOLD of one voxel (cf_index, alpha, tau_ms, w, rho) for one run design.

    Parameters
    ----------
    raw_psth_by_seq : dict {soundfileid: (n_cf, n_bins)}
        Raw population PSTHs of all stimuli, including silence_seq_id (from load_raw_psths).
    silence_seq_id : str
        Key of the silent trial; its mean sharpened rate fills the run's blanks.
    hrf_kernel : np.ndarray, optional
        Built from SUBCORTICAL_PARAMS at 1 ms if None (pass it in when calling repeatedly).

    Returns
    -------
    dict from sweep_run_bold plus full_train (n_samples,).
    """
    assert silence_seq_id in raw_psth_by_seq, f"silent trial '{silence_seq_id}' missing"
    if hrf_kernel is None:
        hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)
    per_seq_train = sharpen_cf_rows(raw_psth_by_seq, cf_index, alpha)
    spont_rate = float(per_seq_train[silence_seq_id].mean())
    full_train = assemble_sweep_run_train(per_seq_train, run_design, total_run_dur_s, spont_rate)
    result = sweep_run_bold(full_train, hrf_kernel, tr_s, w, K, rectify, rho, tau_ms, tau_ms_off)
    result["full_train"] = full_train
    return result
