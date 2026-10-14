"""forward_model_fixed_reference.py
====================================
Same stage ORDER as ``forward_model.py`` (sharpen the full population PSTH
BEFORE CF-selection/chunking — matches CLAUDE.md / full_pipeline_with_adaptrans.py),
but replaces its per-sequence SELF-NORMALIZING rescale with the FIXED,
per-CF reference table from ``reference_calibration.py`` (see project
discussion).

Why this still needed a fix even though forward_model.py's order never
showed the "alpha has literally zero effect" failure mode that
forward_model_torch_order.py did: forward_model.py's existing rescale
(``apply_powerlaw_population``, via ``sharpen_all_sequences``) is computed
FRESH, per sequence, from that ONE sequence's own (n_cf, n_bins) population —
preserving THAT SEQUENCE's own aggregate (across-CF) mean, independently of
every other sequence's own aggregate mean. Individual CF rows can still
shift relative to EACH OTHER within one sequence (that's why alpha produced
real effects there), but each sequence's own OVERALL aggregate level is
still separately self-normalized sequence-by-sequence — the same class of
issue, just one level up (per-sequence instead of per-CF-per-call). Using
one FIXED reference per CF, pooled across every active sequence, removes
this residual self-normalization too, and makes this pipeline directly
comparable to forward_model_torch_order_fixed_reference.py (same
calibration philosophy, only the stage order differs).

Usage: build the reference table ONCE (``reference_calibration.
build_per_cf_reference_table``), then pass it into
``sharpen_all_sequences_fixed_reference`` / ``build_per_seq_trains`` (the
latter reused UNCHANGED from forward_model.py — it only consumes the
already-sharpened population dict, agnostic to how it was produced).
"""

import numpy as np


def apply_powerlaw_population_fixed_reference(
    population_psth: np.ndarray,
    alpha: float,
    reference_table: np.ndarray,
) -> np.ndarray:
    """Power-law sharpen a (n_cf, n_bins) population PSTH, rescaling EACH
    ROW with ITS OWN CF's fixed reference (not one shared constant for the
    whole matrix, and not a fresh rescale from this matrix's own values).

    Parameters
    ----------
    population_psth : np.ndarray, shape (n_cf, n_bins)
        RAW (unsharpened) population PSTH for one sequence.
    alpha : float
    reference_table : np.ndarray, shape (n_cf, N)
        Fixed per-CF calibration reference — see
        ``reference_calibration.build_per_cf_reference_table``.

    Returns
    -------
    np.ndarray, shape (n_cf, n_bins)
    """
    sharpened = population_psth ** alpha
    ref_sharpened = reference_table ** alpha
    in_mean = reference_table.mean(axis=1, keepdims=True)
    out_mean = ref_sharpened.mean(axis=1, keepdims=True)
    scale = np.where(out_mean > 0, in_mean / out_mean, 1.0)
    return sharpened * scale


def sharpen_all_sequences_fixed_reference(
    raw_sequences: dict,
    alpha: float,
    reference_table: np.ndarray,
    silence_seq_id: str,
) -> dict:
    """Fixed-reference counterpart of ``forward_model.sharpen_all_sequences``.

    The silence sequence is passed through UNCHANGED (no sharpening at
    all), not sharpened-then-rescaled like every active sequence — it was
    deliberately excluded from the reference pool (see
    ``reference_calibration.py``), and applying a scale calibrated on
    active data's heavy tail to silence's much narrower raw dynamic range
    caused it to collapse toward zero at high alpha (confirmed empirically:
    silence dropped from 43 to 0.0001 at alpha=8 before this fix, while
    active stayed roughly stable) -- silence was never part of what this
    rescale is supposed to calibrate.

    Returns
    -------
    dict
        ``{seq_id: sharpened_population_psth (n_cf, n_bins)}``
    """
    return {
        seq_id: (
            entry["population_psth"] if seq_id == silence_seq_id else
            apply_powerlaw_population_fixed_reference(entry["population_psth"], alpha, reference_table)
        )
        for seq_id, entry in raw_sequences.items()
    }
