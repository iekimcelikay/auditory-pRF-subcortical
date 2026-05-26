"""timing_utils.py
==================
Shared helpers for formatting and parsing duration/ISI tokens in filenames.

Convention: durations stored as integer milliseconds, e.g. ``dur251ms``.

Filename token format
---------------------
    dur<N>ms   — tone-on duration in ms (integer)
    isi<N>ms   — inter-stimulus interval in ms (integer)

Examples
--------
    >>> fmt_dur_ms(250.88)
    'dur251ms'
    >>> parse_timing("cond01_fc400hz_dur251ms_isi100ms")
    (251.0, 100.0)
"""

import re
from typing import Optional, Tuple

_DUR_RE = re.compile(r"dur(\d+(?:\.\d+)?)ms", re.IGNORECASE)
_ISI_RE = re.compile(r"isi(\d+(?:\.\d+)?)ms", re.IGNORECASE)

COND_ID_RE = re.compile(r"(cond\d+_fc\d+hz_dur\d+ms_isi\d+ms)")


def fmt_dur_ms(dur_ms: float) -> str:
    """Return ``dur<N>ms`` token, rounding to nearest integer."""
    return f"dur{int(round(dur_ms))}ms"


def fmt_isi_ms(isi_ms: float) -> str:
    """Return ``isi<N>ms`` token, rounding to nearest integer."""
    return f"isi{int(round(isi_ms))}ms"


def parse_timing(identifier: str) -> Optional[Tuple[float, float]]:
    """Extract ``(tone_dur_ms, isi_ms)`` from a filename identifier.

    Parameters
    ----------
    identifier : str
        Filename stem containing ``dur<N>ms`` and ``isi<N>ms`` tokens.

    Returns
    -------
    (tone_dur_ms, isi_ms) as floats, or None if tokens are missing.
    """
    dur_match = _DUR_RE.search(identifier)
    isi_match = _ISI_RE.search(identifier)
    if dur_match and isi_match:
        return float(dur_match.group(1)), float(isi_match.group(1))
    return None
