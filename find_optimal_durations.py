"""
Find tone-on durations that fit into a fixed-length sequence.

Formula: seq_dur / N = total_dur  -->  dur = seq_dur/N - ISI

Iterate over candidate N values; print implied dur so you can pick
values close to your target durations.

Usage:
    python find_optimal_durations.py
    python find_optimal_durations.py --isi 50 --seq-dur 20
"""

import argparse
import numpy as np


def compute_durations(
    n_values: list[int],
    isi_ms: float = 50.0,
    seq_dur_s: float = 20.0,
) -> list[dict]:
    """
    For each N, compute the implied tone duration.

    Parameters
    ----------
    n_values : list of int
        Candidate number-of-tones values to evaluate.
    isi_ms : float
        Inter-stimulus interval in ms.
    seq_dur_s : float
        Sequence duration in seconds.

    Returns
    -------
    list of dict with keys: n_tones, total_dur_ms, dur_ms
    """
    seq_dur_ms = seq_dur_s * 1000.0
    results = []
    for n in n_values:
        total_dur_ms = seq_dur_ms / n
        dur_ms = total_dur_ms - isi_ms
        results.append(
            {
                "n_tones": n,
                "total_dur_ms": total_dur_ms,
                "dur_ms": dur_ms,
            }
        )
    return results


def find_closest_durations(
    targets_ms: list[float],
    isi_ms: float = 100.0,
    seq_dur_s: float = 20.0,
    dur_min_ms: float = 10.0,
    dur_max_ms: float = 1000.0,
) -> tuple[float, ...]:
    """
    For each target duration, return the closest achievable float duration (ms).

    Achievable durations are those where N tones of (dur + isi) fit within
    seq_dur with a trailing ISI: N = floor(seq_dur_ms / (dur_ms + isi_ms)).
    Targets above dur_max_ms are clamped so the search stays within range.

    The returned floats are the exact values to pass to the soundgen.
    For filenames / condition keys, callers should use int() (floor) to match
    condition_map convention.

    Parameters
    ----------
    targets_ms : list of float
        Desired tone durations in ms.
    isi_ms : float
        Inter-stimulus interval in ms.
    seq_dur_s : float
        Sequence duration in seconds.
    dur_min_ms, dur_max_ms : float
        Search range for candidate durations.

    Returns
    -------
    tuple of float
        Closest achievable duration (ms) for each target.
    """
    n_values = list(range(10, 10001))
    candidates = compute_durations(n_values, isi_ms=isi_ms, seq_dur_s=seq_dur_s)
    candidates = [r for r in candidates if dur_min_ms <= r["dur_ms"] <= dur_max_ms]

    durs = np.array([r["dur_ms"] for r in candidates])
    result = []
    for t in targets_ms:
        t_clamped = min(float(t), dur_max_ms)  # cap so 500ms target → ≤500ms result
        idx = np.argmin(np.abs(durs - t_clamped))
        result.append(round(float(durs[idx]), 2))
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tone duration implied by N tones in a fixed window.")
    parser.add_argument("--isi",     type=float, default=50.0,  metavar="MS", help="ISI in ms (default: 50)")
    parser.add_argument("--seq-dur", type=float, default=20.0,  metavar="S",  help="Sequence duration in seconds (default: 20)")
    parser.add_argument("--dur-min", type=float, default=20.0,  metavar="MS", help="Min tone duration to show (default: 20)")
    parser.add_argument("--dur-max", type=float, default=500.0, metavar="MS", help="Max tone duration to show (default: 500)")
    parser.add_argument("--groups",  type=int,   default=None,  metavar="N",  help="Pick this many log-spaced representative durations")
    parser.add_argument("--targets", type=float, default=None,  metavar="MS", nargs="+", help="Pick closest valid duration for each target (ms)")
    args = parser.parse_args()

    n_values = list(range(10, 1001))
    results = compute_durations(n_values, isi_ms=args.isi, seq_dur_s=args.seq_dur)
    results = [r for r in results if args.dur_min <= r["dur_ms"] <= args.dur_max]

    print(f"\nISI = {args.isi} ms | seq_dur = {args.seq_dur} s | dur range = [{args.dur_min}, {args.dur_max}] ms\n")
    header = f"{'N (tones)':>10}  {'total_dur (ms)':>15}  {'dur (ms)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['n_tones']:>10}  {r['total_dur_ms']:>15.1f}  {r['dur_ms']:>10.1f}")

    if args.targets:
        durs = np.array([r["dur_ms"] for r in results])
        ns   = np.array([r["n_tones"] for r in results])
        picked = []
        for t in args.targets:
            idx = np.argmin(np.abs(durs - t))
            picked.append({"target_ms": t, "dur_ms": round(float(durs[idx]), 1), "n_tones": int(ns[idx])})

        print(f"\n--- Custom targets ---")
        print(f"{'target (ms)':>12}  {'actual (ms)':>12}  {'N (tones)':>10}")
        print("-" * 38)
        for p in picked:
            print(f"{p['target_ms']:>12.1f}  {p['dur_ms']:>12.1f}  {p['n_tones']:>10}")
        print(f"\nDuration conditions (ms): {[p['dur_ms'] for p in picked]}")
        print(f"N values:                 {[p['n_tones'] for p in picked]}")

    elif args.groups:
        targets = np.logspace(np.log10(args.dur_min), np.log10(args.dur_max), args.groups)
        durs = np.array([r["dur_ms"] for r in results])
        ns   = np.array([r["n_tones"] for r in results])
        picked = []
        for t in targets:
            idx = np.argmin(np.abs(durs - t))
            picked.append({"dur_ms": round(durs[idx], 1), "n_tones": int(ns[idx])})

        print(f"\n--- {args.groups} log-spaced conditions ---")
        print(f"{'N (tones)':>10}  {'dur (ms)':>10}")
        print("-" * 23)
        for p in picked:
            print(f"{p['n_tones']:>10}  {p['dur_ms']:>10.1f}")
        print(f"\nDuration conditions (ms): {[float(p['dur_ms']) for p in picked]}")
        print(f"N values:                 {[p['n_tones'] for p in picked]}")
    else:
        valid_durs = [round(r["dur_ms"], 1) for r in results]
        print(f"\nDuration conditions (ms): {valid_durs}")


# ==============================================================================
# Ramp / tonality helpers  (standalone, not used by main())
# ==============================================================================

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
      1. Ramp fraction per side ≤ max_ramp_fraction.
      2. Sustained portion contains ≥ min_cycles of the carrier.

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


if __name__ == "__main__":
    main()