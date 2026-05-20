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


if __name__ == "__main__":
    main()