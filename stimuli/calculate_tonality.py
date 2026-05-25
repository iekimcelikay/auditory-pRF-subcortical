# calculate_tonality.py
"""
Set of functions to calculate optimal Hz, ramp duration and minimum tone duration for properly perceived tones.
Author: Ekim Celikay
Plomp & Bouman 1959 refers to 'Plomp, R., Bouman, M.A., 1959. Relation between hearing threshold and duration
for tone pulses. J. Acoust. Soc. Am. 31, 749–758'.
"""

import numpy as np

def min_tone_duration_ms(tau_ms: float, max_ramp_fraction: float = 0.10) -> float:
    """Minimum tone duration (ms) for a given ramp not to dominate.

    Parameters
    ----------
    tau_ms : float
        One-sided ramp duration in ms (e.g. TAU_RAMP * 1000).
    max_ramp_fraction : float
        Maximum acceptable fraction of tone duration occupied by one ramp side.
        Default 0.10 (10% each side, 20% total).

    Returns
    -------
    float : minimum tone duration in ms.

    Examples
    --------
    >>> min_tone_duration_ms(5)          # 5 ms ramp, 10% rule
    50.0
    >>> min_tone_duration_ms(5, 0.20)    # 5 ms ramp, 20% rule (more lenient)
    25.0
    """
    return tau_ms / max_ramp_fraction


def max_ramp_ms(tone_dur_ms: float, max_ramp_fraction: float = 0.10) -> float:
    """Maximum ramp duration (ms) for a given tone duration.

    Parameters
    ----------
    tone_dur_ms : float
        Tone-on duration in ms.
    max_ramp_fraction : float
        Maximum acceptable fraction per side. Default 0.10.

    Returns
    -------
    float : maximum one-sided ramp in ms.

    Examples
    --------
    >>> max_ramp_ms(30)   # 30 ms tone, 10% rule → 3 ms max ramp
    3.0
    >>> max_ramp_ms(30, 0.20)
    6.0
    """
    return tone_dur_ms * max_ramp_fraction


def sustained_cycles(tone_dur_ms: float, tau_ms: float, freq_hz: float) -> float:
    """Number of carrier cycles in the flat (post-ramp) portion of a tone.

    Parameters
    ----------
    tone_dur_ms : float
        Total tone-on duration in ms.
    tau_ms : float
        One-sided ramp duration in ms.
    freq_hz : float
        Carrier frequency in Hz.

    Returns
    -------
    float : number of complete cycles in the sustained portion.

    Examples
    --------
    >>> sustained_cycles(30, 5, 400)   # 30ms tone, 5ms ramp, 400 Hz
    8.0
    """
    sustained_ms = tone_dur_ms - 2 * tau_ms
    return max(0.0, sustained_ms / 1000.0 * freq_hz)


def is_tonal(
    tone_dur_ms: float,
    tau_ms: float,
    freq_hz: float,
    min_cycles: float = 4.0,
    max_ramp_fraction: float = 0.20,
) -> bool:
    """Return True if a tone is likely perceived as tonal rather than click-like.

    Applies two checks:
      1. Ramp fraction per side ≤ max_ramp_fraction. "spectral splatter check"
      2. Sustained portion contains ≥ min_cycles of the carrier. "Plomp-Bouman(1959) minimum duration check"

    Parameters
    ----------
    tone_dur_ms : float
        Total tone-on duration in ms.
    tau_ms : float
        One-sided ramp duration in ms.
    freq_hz : float
        Carrier frequency in Hz.
    min_cycles : float
        Minimum carrier cycles in the sustained portion. Default 4.
    max_ramp_fraction : float
        Maximum ramp fraction per side. Default 0.20 (lenient).

    Examples
    --------
    >>> is_tonal(20, 5, 400)   # click — ramp is 25% each side
    False
    >>> is_tonal(30, 5, 400)   # fine — 8 cycles, 16% ramp
    True
    """
    ramp_ok   = (tau_ms / tone_dur_ms) <= max_ramp_fraction
    cycles_ok = sustained_cycles(tone_dur_ms, tau_ms, freq_hz) >= min_cycles
    return ramp_ok and cycles_ok


def tonality_report(
    tone_durs_ms: list,
    tau_ms: float,
    freq_hz: float,
) -> None:
    """Print a tonality check table for a list of tone durations.

    Parameters
    ----------
    tone_durs_ms : list of float
        Tone durations to evaluate (ms).
    tau_ms : float
        One-sided ramp duration in ms.
    freq_hz : float
        Carrier frequency in Hz.
    """
    print(f"\ntau={tau_ms} ms | freq={freq_hz} Hz\n")
    header = f"{'dur (ms)':>10}  {'ramp %':>8}  {'sustained ms':>14}  {'cycles':>8}  {'tonal?':>8}"
    print(header)
    print("-" * len(header))
    for d in tone_durs_ms:
        ramp_pct   = tau_ms / d * 100
        sus_ms     = max(0.0, d - 2 * tau_ms)
        cycles     = sustained_cycles(d, tau_ms, freq_hz)
        tonal      = is_tonal(d, tau_ms, freq_hz)
        flag       = "YES" if tonal else "NO  <--"
        print(f"{d:>10.1f}  {ramp_pct:>7.1f}%  {sus_ms:>13.1f}  {cycles:>8.1f}  {flag}")

