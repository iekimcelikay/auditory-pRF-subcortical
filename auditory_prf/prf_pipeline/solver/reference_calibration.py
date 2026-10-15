"""reference_calibration.py
============================
Builds the pre-power-law POPULATION pool used to rescale the power-law
sharpening (tone-window granularity, i.e. torch order: select CF -> chunk ->
sharpen).

The rescale factor is ``mean(pool) / mean(pool ** alpha)`` — computed per
alpha call, applied to the selected CF's sharpened per-tone mean rates, so the
selected CF is normalized against the whole population instead of against
itself (which pinned its mean and made alpha unable to express any gain
difference across CFs).

Pool contents:
  - every CF x every tone window of every ACTIVE sequence, and
  - the SILENCE response (one mean rate per CF).

The pool is built ONCE from the raw (unsharpened) cochlear data and reused.

``build_per_cf_reference_table`` (the earlier fixed per-CF table) is kept
because ``build_population_pool`` is assembled from it.

Used by the ``kim2024_style_*torch_order*`` recovery scripts through
``forward_model_torch_order_fixed_reference.py``.
"""

import numpy as np

from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id


def build_per_cf_reference_table(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    silence_seq_id: str,
    chunk_margin_ms: float = 30.0,
) -> np.ndarray:
    """Pool raw per-tone mean rates across every ACTIVE sequence, per CF.

    Parameters
    ----------
    raw_sequences : dict
        Output of ``forward_model.load_raw_sequences`` — RAW (unsharpened)
        population PSTH per sequence.
    cf_list_hz : np.ndarray, shape (n_cf,)
    silence_seq_id : str
        Excluded from the pool — see module docstring.
    chunk_margin_ms : float
        Passed through to ``chunk_from_id``, matching whatever margin the
        calling pipeline uses elsewhere (keep consistent with it).

    Returns
    -------
    np.ndarray, shape (n_cf, N)
        N = total tone-pips pooled across every active sequence (same for
        every CF, since every CF sees the same set of sequences/chunks —
        rectangular by construction, no padding needed).
    """
    n_cf = len(cf_list_hz)
    per_cf_rates: list = [[] for _ in range(n_cf)]

    for seq_id, entry in raw_sequences.items():
        if seq_id == silence_seq_id:
            continue
        time_axis_s = entry["time_axis_s"]
        population_psth = entry["population_psth"]
        for cf_index in range(n_cf):
            raw_row = population_psth[cf_index, :]
            result, _tone_dur_ms, _isi_ms = chunk_from_id(
                raw_row, time_axis_s, seq_id, margin_ms=chunk_margin_ms,
            )
            per_cf_rates[cf_index].extend(float(np.mean(c)) for c in result["chunks"])

    return np.array(per_cf_rates, dtype=np.float64)


def _silence_per_cf_means(
    raw_sequences: dict,
    silence_seq_id: str,
) -> np.ndarray:
    """Mean raw rate of every CF during silence. Returns shape (n_cf,)."""
    return np.asarray(raw_sequences[silence_seq_id]["population_psth"]).mean(axis=1)


def build_population_pool(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    silence_seq_id: str,
    chunk_margin_ms: float = 30.0,
) -> np.ndarray:
    """Pre-power-law POPULATION pool for the power-law rescale (tone-window
    granularity, matching sharpening AFTER chunking).

    Pools the tone-window mean rates of ALL CFs across every active sequence,
    plus the silence response (one mean per CF). Built ONCE; every forward call
    then uses ``mean(pool) / mean(pool ** alpha)`` as the rescale factor.

    Returns
    -------
    np.ndarray, shape (n_cf * n_active_tones + n_cf,)
    """
    active_table = build_per_cf_reference_table(
        raw_sequences, cf_list_hz, silence_seq_id, chunk_margin_ms,
    )                                                     # (n_cf, N)
    silence_means = _silence_per_cf_means(raw_sequences, silence_seq_id)  # (n_cf,)
    assert silence_means.shape[0] == active_table.shape[0]
    return np.concatenate([active_table.ravel(), silence_means])
