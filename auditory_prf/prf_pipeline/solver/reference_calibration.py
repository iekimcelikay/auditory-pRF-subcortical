"""reference_calibration.py
============================
Shared utility for building a FIXED, per-CF power-law-sharpening reference
table — the calibration anchor that replaces each pipeline's previous
SELF-NORMALIZING rescale (which recomputed ``mean(input) == mean(output)``
fresh from whatever narrow slice of data happened to be in scope for one
call, forcing that call's own mean to stay pinned to itself and making alpha
structurally incapable of expressing gain differences across sequences/CFs
— see project discussion).

Design (per project discussion):
  - Per-CF, not global-across-CF: different CFs have genuinely different
    typical response levels (tonotopic tuning is real signal, not noise to
    average away), so one single population-wide constant would distort
    that. Each CF gets its own reference.
  - Pooled across EVERY active sequence, unweighted: no frequency-proximity
    cherry-picking of "preferred" conditions — that would just be a
    different hidden parameter choice (how close counts as close?). A
    sharply-tuned CF ending up with a lower reference than a broadly-tuned
    one is not a bias to correct for; it's an honest reflection of what
    "sharply tuned" means when averaged over its whole natural stimulus
    diet.
  - Silence/null excluded: the reference calibrates the duration/CF-tuned
    nonlinearity specifically, on active (tone-driven) trials only.
  - Computed ONCE, offline, from the raw (unsharpened) cochlear data — never
    recomputed from a single forward call's own narrow input.

Used by:
  - forward_model_fixed_reference.py (sharpen-before-CF-select order,
    matching forward_model.py's existing stage order, but with THIS fixed
    reference replacing its current per-sequence self-normalizing rescale)
  - forward_model_torch_order_fixed_reference.py (CF-select/interpolate-
    then-chunk-then-sharpen order, matching torch_pipeline.py's order, with
    this same fixed reference replacing forward_model_torch_order.py's
    self-normalizing rescale)
  - torch_pipeline.py itself (exported as a torch buffer, interpolated via
    the same CFSelector._interp used for the live signal — see project
    discussion; not yet wired in, pending a separate fix to that file's
    currently-broken Greenwood constants)
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


def build_per_cf_raw_reference_table(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    silence_seq_id: str,
) -> np.ndarray:
    """Pool RAW (unchunked) per-time-bin values across every ACTIVE
    sequence, per CF — same philosophy as ``build_per_cf_reference_table``,
    but at the RAW time-sample granularity instead of chunked tone-pip
    means.

    This granularity match matters: ``forward_model_fixed_reference.py``
    sharpens the FULL, un-averaged ``population_psth`` directly (matching
    forward_model.py's own order), so its reference must be built from
    equally-fine-grained raw samples — using the CHUNKED, already-averaged
    reference from ``build_per_cf_reference_table`` there mismatches
    granularity (x**alpha on raw, spiky samples has much more dynamic range
    than x**alpha on already-averaged chunk means), which caused the
    rescale to massively under-correct and blow up (confirmed empirically:
    >300,000x inflation at alpha=8 before this fix). Every sequence
    contributes the same number of time bins (fixed SIGNAL_DT_S
    resolution), so this is rectangular by construction.

    Returns
    -------
    np.ndarray, shape (n_cf, N)
        N = total raw time-bins pooled across every active sequence's full
        duration (tone-ON periods AND inter-tone gaps alike — matching
        exactly what ``apply_powerlaw_population`` was always calibrated
        against: the whole raw population, not a tone-ON-only subset).
    """
    n_cf = len(cf_list_hz)
    rows = []
    for cf_index in range(n_cf):
        values = []
        for seq_id, entry in raw_sequences.items():
            if seq_id == silence_seq_id:
                continue
            values.append(entry["population_psth"][cf_index, :])
        rows.append(np.concatenate(values))
    return np.stack(rows, axis=0)
