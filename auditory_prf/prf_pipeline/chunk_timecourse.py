"""chunk_timecourse.py
====================
Tone-ON window extraction for the auditory pRF pipeline.

All timing inputs and outputs are in **milliseconds**.
The PSTH bin width (dt_ms) is inferred automatically from the time_axis
stored in the .npz file, so the function works for any bin resolution.

----------
parse_tone_timing(identifier)          -> (tone_dur_ms, isi_ms) | None
compute_tone_onsets_offsets(...)       -> (onsets_ms, offsets_ms)
chunk_timecourse(...)                  -> dict
"""

from __future__ import annotations
import re
import numpy as np
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def parse_tone_timing(identifier: str) -> Optional[Tuple[float, float]]:
    """Extract ``(tone_dur_ms, isi_ms)`` from a filename identifier.

    Expects tokens of the form ``dur<N>ms`` and ``isi<N>ms`` anywhere in the
    identifier string (e.g.
    ``dipc_sequence03_fc125hz_dur267ms_isi67ms_...``).
    Returns ``None`` if either token is not found.
    """
    dur_match = re.search(r"dur(\d+(?:\.\d+)?)ms", identifier, re.IGNORECASE)
    isi_match = re.search(r"isi(\d+(?:\.\d+)?)ms", identifier, re.IGNORECASE)
    if dur_match and isi_match:
        return float(dur_match.group(1)), float(isi_match.group(1))
    return None


def compute_tone_onsets_offsets(
    tone_dur_ms: float,
    isi_ms: float,
    total_dur_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute tone onset and offset times for a periodic tone sequence.

    Uses the same formula as ``stimuli/soundgen.py::calculate_num_tones``:

        num_tones = int((total_dur_ms + isi_ms) / (tone_dur_ms + isi_ms))

    Parameters
    ----------
    tone_dur_ms : float
        Duration of each tone in ms.
    isi_ms : float
        Inter-stimulus interval (silence between tones) in ms.
    total_dur_ms : float
        Total sequence duration in ms.

    Returns
    -------
    onsets_ms : ndarray, shape (num_tones,)
    offsets_ms : ndarray, shape (num_tones,)
        Both in ms.
    """
    period_ms = tone_dur_ms + isi_ms
    num_tones = int((total_dur_ms + isi_ms) / period_ms)
    onsets_ms  = np.arange(num_tones, dtype=float) * period_ms
    offsets_ms = onsets_ms + tone_dur_ms
    return onsets_ms, offsets_ms


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_timecourse(
    timecourse: np.ndarray,
    time_axis_s: np.ndarray,
    tone_dur_ms: float,
    isi_ms: float,
    total_dur_ms: float,
    margin_ms: float = 50.0,
) -> dict:
    """Cut a timecourse into per-tone chunks (tone ON window + margin).

    Parameters
    ----------
    timecourse : ndarray, shape (n_bins,)
        Firing rate values (e.g. sp/s) at each time bin.
    time_axis_s : ndarray, shape (n_bins,)
        Time axis in **seconds** as stored in the .npz file.
    tone_dur_ms : float
        Duration of each tone in ms.
    isi_ms : float
        Inter-stimulus interval in ms.
    total_dur_ms : float
        Total sequence duration in ms.
    margin_ms : float, optional
        Extra window after tone offset in ms (default 50 ms).

    Returns
    -------
    dict with keys:
        chunks       : List[ndarray]  — firing rate values per tone window
        axes_abs_ms  : List[ndarray]  — absolute time in ms per window
        axes_rel_ms  : List[ndarray]  — time in ms relative to tone onset
        onsets_ms    : ndarray, shape (num_tones,)
        offsets_ms   : ndarray, shape (num_tones,)
        dt_ms        : float  — bin width in ms, inferred from time_axis_s
    """
    # -- infer bin width and convert time axis to ms -------------------------
    dt_ms        = (time_axis_s[1] - time_axis_s[0]) * 1000.0
    time_axis_ms = time_axis_s * 1000.0

    # tolerance: half a bin width guards against floating-point drift
    atol = dt_ms * 0.5

    # -- compute tone timing -------------------------------------------------
    onsets_ms, offsets_ms = compute_tone_onsets_offsets(
        tone_dur_ms, isi_ms, total_dur_ms
    )

    # -- slice per tone ------------------------------------------------------
    chunks: List[np.ndarray]      = []
    axes_abs_ms: List[np.ndarray] = []
    axes_rel_ms: List[np.ndarray] = []

    for onset_ms, offset_ms in zip(onsets_ms, offsets_ms):
        window_end_ms = offset_ms + margin_ms
        mask = (
            (time_axis_ms >= onset_ms - atol) &
            (time_axis_ms <  window_end_ms + atol)
        )
        chunks.append(timecourse[mask])
        axes_abs_ms.append(time_axis_ms[mask])
        axes_rel_ms.append(time_axis_ms[mask] - onset_ms)

    return {
        "chunks":      chunks,
        "axes_abs_ms": axes_abs_ms,
        "axes_rel_ms": axes_rel_ms,
        "onsets_ms":   onsets_ms,
        "offsets_ms":  offsets_ms,
        "dt_ms":       dt_ms,
    }

def chunk_from_id(
        timecourse: np.ndarray,
        time_axis_s: np.ndarray,
        identifier: str,
        margin_ms: float = 50.0,
    ) -> Tuple[dict, float, float]:
    """ Parse tone timign from identifier and chunk timecourse.

    Wraps ``parse_tone_timing``, ``total_dur_ms`` computation, and
    ``chunk_timecourse`` into a single call for use in the pipeline.

    Parameters
    ----------
    timecourse : ndarray, shape (n_bins,)
        Firing rate values after power-law sharpening.
    time_axis_s: ndarray, shape (n_bins,)
        Time axis ins seconds as stored in the .npz file.
    identifier: str
        Filename or seq_id containing ``dur<N>ms`` and ``isi<N>ms`` tokens.
    margin_ms : float, optional
        Extra window after tone offset in ms (default 50 ms).

    Returns
    -------
    result: dict
        Full output from ``chunk_timecourse`` (chunks, axes, onsets, offsets, dt_ms).
    tone_dur_ms : float
        Parsed tone duration in ms.
    isi_ms: float
        Parsed ISI in ms.

    Raises
    ------
    ValueError
        If tone timing tokens cannot be parsed from ``identifier``.

    """
    # NOTE: FUNCTION TO BE USED IN PIPELINE SCRIPT
    # TODO: TEST IT AND VISUALIZE IT
    timing = parse_tone_timing(identifier)
    if timing is None:
        raise ValueError(
            f"Could not parse tone timing from identifier: {identifier!r}")

    tone_dur_ms, isi_ms = timing

    dt_ms = (time_axis_s[1] - time_axis_s[0]) * 1000.0  #  get the dt (difference btw sample points in time)
    total_dur_ms = time_axis_s[-1] * 1000.0 + dt_ms

    result = chunk_timecourse(
        timecourse, time_axis_s, tone_dur_ms, isi_ms, total_dur_ms, margin_ms)

    return result, tone_dur_ms, isi_ms


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    NPZ_PATH = (
        "/home/ekim/auditory-pRF-subcortical/models_output/"
        "dipc_sequence03_fc125hz_dur267ms_isi133ms_adapted_timecourse.npz"
    )

    # -- load ----------------------------------------------------------------
    data        = np.load(NPZ_PATH, allow_pickle=True)
    time_axis_s = data["time_axis"]          # seconds
    timecourse  = data["timecourse"]         # sp/s, shape (n_bins,)

    # -- parse timing from filename ------------------------------------------
    identifier = os.path.basename(NPZ_PATH)
    timing = parse_tone_timing(identifier)
    if timing is None:
        raise ValueError(f"Could not parse tone timing from: {identifier}")
    tone_dur_ms, isi_ms = timing

    # total duration derived from the time axis itself — no hardcoding
    dt_ms         = (time_axis_s[1] - time_axis_s[0]) * 1000.0
    total_dur_ms  = time_axis_s[-1] * 1000.0 + dt_ms

    print(f"tone_dur_ms  : {tone_dur_ms} ms")
    print(f"isi_ms       : {isi_ms} ms")
    print(f"total_dur_ms : {total_dur_ms:.2f} ms")
    print(f"dt_ms        : {dt_ms:.4f} ms")

    # -- chunk ---------------------------------------------------------------
    result = chunk_timecourse(
        timecourse   = timecourse,
        time_axis_s  = time_axis_s,
        tone_dur_ms  = tone_dur_ms,
        isi_ms       = isi_ms,
        total_dur_ms = total_dur_ms,
        margin_ms    = 50.0,
    )

    # -- inspect -------------------------------------------------------------
    print(f"\nnum tones    : {len(result['chunks'])}")
    print(f"onsets_ms    : {result['onsets_ms']}")
    print(f"offsets_ms   : {result['offsets_ms']}")
    for i, (chunk, ax_rel) in enumerate(
        zip(result["chunks"], result["axes_rel_ms"])
    ):
        print(
            f"  tone {i}: {len(chunk)} bins | "
            f"rel axis [{ax_rel[0]:.2f} .. {ax_rel[-1]:.2f}] ms | "
            f"mean FR = {chunk.mean():.2f} sp/s"
        )


