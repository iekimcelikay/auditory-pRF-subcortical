"""
Find tone-on durations that fit perfectly into a fixed-length sequence.

A duration is "optimal" when (tone_dur + isi) evenly divides seq_len_ms,
so no partial tone is left at the end of the window.

Usage:
    python find_optimal_durations.py
    python find_optimal_durations.py --isi 67 --seq-len 20000 --dur-min 20 --dur-max 500
"""

import argparse


def find_valid_durations(
    isi_ms: float = 100.0,
    seq_len_ms: float = 20000.0,
    dur_min_ms: float = 20.0,
    dur_max_ms: float = 500.0,
) -> list[dict]:
    """
    Return durations where (dur + isi) divides seq_len_ms exactly.

    Parameters
    ----------
    isi_ms : float
        Inter-stimulus interval in ms.
    seq_len_ms : float
        Total sequence window length in ms.
    dur_min_ms : float
        Minimum tone duration to consider (ms).
    dur_max_ms : float
        Maximum tone duration to consider (ms).

    Returns
    -------
    list of dict, each with keys: dur_ms, period_ms, n_tones, duty_cycle
    """
    results = []
    for dur in range(int(dur_min_ms), int(dur_max_ms) + 1):
        period = dur + isi_ms
        if seq_len_ms % period == 0:
            n_tones = int(seq_len_ms / period)
            results.append(
                {
                    "dur_ms": dur,
                    "period_ms": period,
                    "n_tones": n_tones,
                    "duty_cycle": dur / period,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Find tone durations that tile a sequence window exactly.")
    parser.add_argument("--isi",     type=float, default=100.0,   metavar="MS",  help="ISI in ms (default: 100)")
    parser.add_argument("--seq-len", type=float, default=20000.0, metavar="MS",  help="Sequence length in ms (default: 20000)")
    parser.add_argument("--dur-min", type=float, default=20.0,    metavar="MS",  help="Min tone duration in ms (default: 20)")
    parser.add_argument("--dur-max", type=float, default=500.0,   metavar="MS",  help="Max tone duration in ms (default: 500)")
    args = parser.parse_args()

    hits = find_valid_durations(
        isi_ms=args.isi,
        seq_len_ms=args.seq_len,
        dur_min_ms=args.dur_min,
        dur_max_ms=args.dur_max,
    )

    print(f"\nISI = {args.isi} ms | seq_len = {args.seq_len} ms | "
          f"dur range = [{args.dur_min}, {args.dur_max}] ms\n")

    if not hits:
        print("No valid durations found.")
        return

    header = f"{'dur (ms)':>10}  {'period (ms)':>12}  {'n_tones':>8}  {'duty_cycle':>11}"
    print(header)
    print("-" * len(header))
    for r in hits:
        print(f"{r['dur_ms']:>10}  {r['period_ms']:>12.1f}  {r['n_tones']:>8}  {r['duty_cycle']:>11.3f}")

    durs = [r["dur_ms"] for r in hits]
    print(f"\nFound {len(hits)} valid duration(s): {durs}")
    print(f"As tuple: {tuple(durs)}")


if __name__ == "__main__":
    main()