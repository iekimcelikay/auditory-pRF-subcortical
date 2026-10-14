"""forward_model_torch_order_fixed_reference.py
=================================================
Same stage ORDER as ``forward_model_torch_order.py`` (CF-select first, chunk
into tone-ON windows, THEN sharpen — matching ``torch_pipeline.py``'s
AuditoryPRFPipeline.forward() order, which is biologically correct here:
the alpha nonlinearity models a SUBCORTICAL process acting after CF-pooling,
not something happening separately in each peripheral cochlear channel
before they converge — see project discussion), but replaces its
SELF-NORMALIZING rescale (computed fresh from each call's own single-
sequence (n_tones,) vector, which made alpha structurally incapable of
expressing any gain difference across sequences/CFs) with the FIXED,
per-CF reference table from ``reference_calibration.py``.

No interpolation needed here (unlike torch_pipeline.py): this numpy
pipeline uses a discrete ``cf_index``, so the reference for a given CF is
just ``reference_table[cf_index, :]`` directly — no blending between
neighboring rows.
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
    reference_table: np.ndarray,
    chunk_margin_ms: float = 30.0,
    pref_dur_ms: Optional[float] = None,
    sigma_dur_ms: float = 50.0,
) -> tuple[dict, float]:
    """Build the boxcar-train ``per_seq`` dict for one (cf_index, alpha)
    pair, chunking BEFORE sharpening (torch_pipeline.py's order), using a
    FIXED per-CF reference for the rescale instead of a fresh self-
    normalizing one.

    Parameters
    ----------
    reference_table : np.ndarray, shape (n_cf, N)
        Fixed per-CF calibration reference — see
        ``reference_calibration.build_per_cf_reference_table``. Computed
        ONCE, reused across every (cf_index, alpha) call — NOT derived from
        this call's own data.
    Other parameters: see ``forward_model_torch_order.build_per_seq_trains_torch_order``.

    Returns
    -------
    per_seq : dict
        ``{seq_id: {"train": np.ndarray}}``
    cf_hz : float
    """
    cf_hz = float(cf_list_hz[cf_index])

    reference_for_this_cf = reference_table[cf_index, :]
    ref_in_mean = reference_for_this_cf.mean()
    ref_out_mean = (reference_for_this_cf ** alpha).mean()
    scale = ref_in_mean / ref_out_mean if ref_out_mean > 0 else 1.0

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

            # Sharpen AFTER chunking (CF already selected, matching
            # torch_pipeline.py's order), rescaled with the FIXED per-CF
            # reference computed across every active sequence -- NOT this
            # one sequence's own mean.
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
