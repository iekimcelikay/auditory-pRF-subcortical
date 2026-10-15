"""forward_model_torch_order.py
================================
Alternate per-sequence forward model matching ``torch_pipeline.py``'s stage
order, kept as a SEPARATE module from ``forward_model.py`` (per project
discussion) so both orderings can be run and compared side by side rather
than overwriting the existing (CLAUDE.md-documented, numpy-pipeline)
behavior in place.

Order difference (see project discussion — confirmed via direct code
inspection of both pipelines):

  forward_model.py (matches CLAUDE.md / full_pipeline_with_adaptrans.py):
    load raw population PSTH -> power-law sharpen the WHOLE population PSTH
    (all CFs x all time bins, one GLOBAL mean-preserving rescale per alpha,
    shared across every cf_index) -> select cf_index row -> chunk into
    tone-ON windows -> per-tone mean rates -> duration filter -> boxcar.

  torch_pipeline.py's AuditoryPRFPipeline.forward() (lines 598-605):
    load raw population PSTH -> select cf_index row (RAW, unsharpened) ->
    chunk into tone-ON windows -> per-tone mean rates -> power-law sharpen
    THIS (n_tones,) mean-rate vector (a LOCAL mean-preserving rescale,
    computed only over this one cf_index's own trial means, not globally
    across the population) -> duration filter -> boxcar.

Since x**alpha is nonlinear, these two orderings are NOT equivalent:
mean(sharpen(full_timecourse)) != sharpen(mean(full_timecourse)) in general,
and the rescale normalization differs (global-population vs. per-CF-local).
This module implements the torch_pipeline.py order for the numpy solver
scripts, reusing ``apply_powerlaw_population`` unchanged (already shape-
generic over any ndarray, so no new sharpening math needed — only WHEN/WHAT
it's called on changes) and ``chunk_from_id``/``build_prf_boxcar_train``/
``apply_duration_gaussian_scalar`` unchanged from forward_model.py's own
dependencies.

IMPORTANT caching consequence for any solver/recovery script built on this
module: in forward_model.py's order, sharpening depends on alpha ALONE, so
``sharpen_all_sequences(raw_sequences, alpha)`` is computed ONCE per alpha
and reused across every cf_index (the whole point of grid_recovery.py's
per-alpha-pool candidate-grid-build structure). In THIS (torch) order,
sharpening depends on (cf_index, alpha) JOINTLY — it happens after
cf_index-specific chunking — so there is NO cross-cf_index sharing: every
(cf_index, alpha) combination needs its own chunk-then-sharpen pass. Any
candidate-grid builder using this module should loop/cache accordingly
(e.g. per-cf_index-and-alpha, not per-alpha-only).
"""

from typing import Optional

import numpy as np

from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train


def build_per_seq_trains_torch_order(
    raw_sequences: dict,
    cf_index: int,
    cf_list_hz: np.ndarray,
    alpha: float,
    silence_seq_id: str,
    chunk_margin_ms: float = 30.0,
    pref_dur_ms: Optional[float] = None,
    sigma_dur_ms: float = 50.0,
) -> tuple[dict, float]:
    """Build the boxcar-train ``per_seq`` dict for one (cf_index, alpha) pair,
    sharpening AFTER cf-selection and chunking (torch_pipeline.py's order) —
    contrast with ``forward_model.build_per_seq_trains``, which takes
    ALREADY-sharpened (alpha-only-cached) population data and only varies
    cf_index.

    Parameters
    ----------
    raw_sequences : dict
        Output of ``forward_model.load_raw_sequences`` — RAW (unsharpened)
        population PSTH per sequence.
    cf_index : int
        Zero-based row index into each sequence's population PSTH.
    cf_list_hz : np.ndarray, shape (n_cf,)
    alpha : float
        Power-law sharpening exponent — applied here to this cf_index's own
        chunked per-tone mean-rate vector, not to the full population PSTH.
    silence_seq_id : str
        seq_id of the silence/null-trial stimulus. Silence has no tone-ON
        windows to chunk, so there is nothing for the sharpening step to
        operate on per-trial — matches forward_model.py's own silence
        handling (flat train at the raw row's mean; sharpening a single
        scalar via a mean-preserving rescale is a no-op regardless of order,
        so this is not a point of divergence between the two pipelines).
    chunk_margin_ms : float
        Extra window after tone offset passed to ``chunk_from_id`` (ms).
    pref_dur_ms, sigma_dur_ms : see forward_model.build_per_seq_trains.

    Returns
    -------
    per_seq : dict
        ``{seq_id: {"train": np.ndarray}}``, ready for ``assemble_run_bold``.
    cf_hz : float
    """
    cf_hz = float(cf_list_hz[cf_index])
    per_seq: dict = {}

    for seq_id, entry in raw_sequences.items():
        time_axis_s = entry["time_axis_s"]
        raw_row = entry["population_psth"][cf_index, :]

        dt_s = time_axis_s[1] - time_axis_s[0]
        total_dur_ms = (time_axis_s[-1] + dt_s) * 1000.0

        if seq_id == silence_seq_id:
            train = np.full(int(round(total_dur_ms)), float(np.mean(raw_row)))
        else:
            result, tone_dur_ms, _isi_ms = chunk_from_id(
                raw_row, time_axis_s, seq_id, margin_ms=chunk_margin_ms,
            )
            mean_rates_on = np.array([np.mean(chunk) for chunk in result["chunks"]])

            # Sharpen AFTER chunking, on this (n_tones,) mean-rate vector --
            # a LOCAL mean-preserving rescale (computed only over these
            # ~24-32 trial means), matching torch_pipeline.py's
            # PowerLawSharpening call site exactly (not the GLOBAL,
            # population-wide rescale forward_model.py uses).
            sharpened_rates = apply_powerlaw_population(mean_rates_on, alpha)

            if pref_dur_ms is not None:
                amplitudes = [
                    apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur_ms, sigma_dur_ms)
                    for m in sharpened_rates
                ]
            else:
                amplitudes = list(sharpened_rates)

            train = build_prf_boxcar_train(
                amplitudes,
                result["onsets_ms"],
                result["offsets_ms"],
                total_dur_ms,
                dt_ms=1.0,
            )
        per_seq[seq_id] = {"train": train}

    return per_seq, cf_hz
