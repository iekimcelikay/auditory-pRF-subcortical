"""forward_model_torch_order_fixed_reference.py
=================================================
Same stage ORDER as ``forward_model_torch_order.py`` (CF-select first, chunk
into tone-ON windows, THEN sharpen — matching ``torch_pipeline.py``'s
AuditoryPRFPipeline.forward() order, which is biologically correct here:
the alpha nonlinearity models a SUBCORTICAL process acting after CF-pooling,
not something happening separately in each peripheral cochlear channel
before they converge — see project discussion), but replaces its
SELF-NORMALIZING rescale (computed fresh from the selected CF's own
single-sequence (n_tones,) vector, which pinned that CF to itself) with a
POPULATION normalization: the rescale factor is
``mean(pool) / mean(pool ** alpha)``, where ``pool`` is the pre-power-law
population pool from ``reference_calibration.build_population_pool`` (all CFs
x all tone windows of every active sequence, PLUS the silence response).
The pool is built once and passed in; the factor is recomputed per call
because the post-sharpening side depends on alpha. It is applied to the
selected CF's sharpened rates, and to the silence train as well.
"""

from typing import Optional

import numpy as np

from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train


def build_per_seq_trains_torch_order_fixed_reference(
    raw_sequences: dict,
    cf_index: int,
    cf_list_hz: np.ndarray,
    alpha: float,
    silence_seq_id: str,
    population_pool: np.ndarray,
    chunk_margin_ms: float = 30.0,
    pref_dur_ms: Optional[float] = None,
    sigma_dur_ms: float = 50.0,
) -> tuple[dict, float]:
    """Build the boxcar-train ``per_seq`` dict for one (cf_index, alpha)
    pair, chunking BEFORE sharpening (torch_pipeline.py's order), rescaling
    with the pre-power-law population pool.

    Parameters
    ----------
    population_pool : np.ndarray, shape (n_pool,)
        Pre-power-law population rates incl. silence — see
        ``reference_calibration.build_population_pool``. Computed ONCE,
        reused across every (cf_index, alpha) call.
    Other parameters: see ``forward_model_torch_order.build_per_seq_trains_torch_order``.

    Returns
    -------
    per_seq : dict
        ``{seq_id: {"train": np.ndarray}}``
    cf_hz : float
    """
    cf_hz = float(cf_list_hz[cf_index])

    assert population_pool.ndim == 1 and population_pool.size > 0, population_pool.shape
    pool_out_mean = (population_pool ** alpha).mean()
    scale = population_pool.mean() / pool_out_mean if pool_out_mean > 0 else 1.0

    per_seq: dict = {}

    for seq_id, entry in raw_sequences.items():
        time_axis_s = entry["time_axis_s"]
        raw_row = entry["population_psth"][cf_index, :]

        dt_s = time_axis_s[1] - time_axis_s[0]
        total_dur_ms = (time_axis_s[-1] + dt_s) * 1000.0

        if seq_id == silence_seq_id:
            # Silence is part of the population pool, so it is sharpened and
            # rescaled with the same factor as the active sequences.
            silence_rate = float(np.mean(raw_row)) ** alpha * scale
            train = np.full(int(round(total_dur_ms)), silence_rate)
        else:
            result, tone_dur_ms, _isi_ms = chunk_from_id(
                raw_row, time_axis_s, seq_id, margin_ms=chunk_margin_ms,
            )
            mean_rates_on = np.array([np.mean(chunk) for chunk in result["chunks"]])

            # Sharpen AFTER chunking (CF already selected, matching
            # torch_pipeline.py's order), rescaled with the population-pool
            # factor -- NOT this CF's or this sequence's own mean.
            sharpened_rates = (mean_rates_on ** alpha) * scale

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
