"""
rank_run_designs.py
===================
Rank 1000+ run designs by normalised distance from mean predicted BOLD,
aggregated across active CFs.

"Best" designs produce BOLD closest to the cross-design average —
the most representative orderings for pRF fitting.

Outputs
-------
design_ranking.npz   — full ranked list (all designs)
design_ranking.csv   — trial-level structure for the top/bottom N designs

Usage
-----
    python visualization/scratch/rank_run_designs.py
    python visualization/scratch/rank_run_designs.py --active_cfs 7 8 9 10 11 --top_n 20
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from auditory_prf.utils.stimulus_utils import calc_cfs
from auditory_prf.prf_pipeline.run_assembly import make_seq_id_fn, generate_run_design

# ── must match run_pipeline_notemporal.py ─────────────────────────────────────
BASE_SEED            = 42
ALL_DURATIONS        = (35, 45, 60, 75, 100, 150, 251, 488)
ISI_MS               = 100
FREQ_RANGE           = (450, 1600, 3)
TRIAL_DURATION_S     = 20.0
OPENING_BLANK_S      = 10.0
ITI_RANGE_S          = (0, 0)
NULL_FRACTION        = 0.25
N_CONDITIONS_PER_RUN = len(ALL_DURATIONS)   # all durations every run

ACTIVE_CFS = list(range(7, 19))


# ── loading ───────────────────────────────────────────────────────────────────

def load_bold_matrix(npz_path: Path) -> tuple:
    d = np.load(npz_path, allow_pickle=True)
    run_keys = [k for k in d.files if k.startswith("run_")]
    run_keys.sort(key=lambda k: int(k.split("_")[1]))
    run_numbers = [int(k.split("_")[1]) for k in run_keys]
    bold_matrix = np.stack([d[k] for k in run_keys], axis=0)
    return bold_matrix, run_numbers, int(d["cf"])


# ── ranking ───────────────────────────────────────────────────────────────────

def distance_from_mean(bold_matrix: np.ndarray) -> np.ndarray:
    """Mean squared z-score deviation of each run from the cross-design mean.

    Parameters
    ----------
    bold_matrix : np.ndarray, shape (n_runs, n_trs)

    Returns
    -------
    dist : np.ndarray, shape (n_runs,)
        Lower = closer to mean = more representative design.
    """
    mean_bold = bold_matrix.mean(axis=0)
    std_bold  = bold_matrix.std(axis=0)
    std_bold  = np.where(std_bold > 0, std_bold, 1.0)
    z = (bold_matrix - mean_bold) / std_bold
    return (z ** 2).mean(axis=1)


# ── design reconstruction ─────────────────────────────────────────────────────

def _build_helpers():
    desired_freqs = calc_cfs(FREQ_RANGE, species='human')
    seq_id_fn     = make_seq_id_fn(FREQ_RANGE,
                                   tone_on_ms_options=ALL_DURATIONS,
                                   isi_ms_options=(ISI_MS,) * len(ALL_DURATIONS))
    return desired_freqs, seq_id_fn


def reconstruct_run_design(run_number: int, desired_freqs, seq_id_fn) -> list:
    """Replay the deterministic design generation for one run number.

    Parameters
    ----------
    run_number : int — 1-based run number (matches npz key suffix)

    Returns
    -------
    list of (seq_id, onset_s)
    """
    i   = run_number - 1
    rng = np.random.default_rng(BASE_SEED + i)

    sampled_durations = rng.choice(ALL_DURATIONS, size=N_CONDITIONS_PER_RUN, replace=False)

    stimuli     = [(int(dur), ISI_MS, freq)
                   for dur in sampled_durations
                   for freq in desired_freqs]
    n_null      = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
    base_trials = stimuli + [(0, 0, None)] * n_null
    design_seed = int(rng.integers(0, 2**32))

    return generate_run_design(
        base_trials, seq_id_fn,
        trial_duration_s = TRIAL_DURATION_S,
        opening_blank_s  = OPENING_BLANK_S,
        iti_range_s      = ITI_RANGE_S,
        seed             = design_seed,
    )


def parse_seq_id(seq_id) -> tuple:
    """Extract (freq_hz, duration_ms, isi_ms) from a seq_id string, or nulls."""
    if seq_id is None or seq_id == "null" or seq_id.startswith("cond00"):
        return None, None, None
    freq_hz     = int(re.search(r'_fc(\d+)hz',  seq_id).group(1))
    duration_ms = int(re.search(r'_dur(\d+)ms', seq_id).group(1))
    isi_ms      = int(re.search(r'_isi(\d+)ms', seq_id).group(1))
    return freq_hz, duration_ms, isi_ms


# ── csv export ────────────────────────────────────────────────────────────────

def _design_rows(ranked_run_numbers, distances, top_n, desired_freqs, seq_id_fn):
    """Reconstruct trial sequences for best and worst designs."""
    groups = [
        ("best",  ranked_run_numbers[:top_n],        range(1, top_n + 1)),
        ("worst", ranked_run_numbers[-top_n:][::-1], range(1, top_n + 1)),
    ]
    for group, run_nums, ranks in groups:
        for rank, run_num in zip(ranks, run_nums):
            dist   = distances[np.where(ranked_run_numbers == run_num)[0][0]]
            design = reconstruct_run_design(run_num, desired_freqs, seq_id_fn)
            trials = []
            for seq_id, onset_s in design:
                freq_hz, duration_ms, _ = parse_seq_id(seq_id)
                condition = f"{freq_hz}Hz_{duration_ms}ms" if freq_hz is not None else "null"
                trials.append((round(float(onset_s), 3), condition))
            yield group, rank, run_num, round(float(dist), 6), trials


def save_long_csv(ranked_run_numbers, distances, top_n: int,
                  desired_freqs, seq_id_fn, out_path: Path) -> None:
    """One row per trial: group, rank, run_number, trial_idx, onset_s, condition."""
    rows = []
    for group, rank, run_num, dist, trials in _design_rows(
            ranked_run_numbers, distances, top_n, desired_freqs, seq_id_fn):
        for trial_idx, (onset_s, condition) in enumerate(trials, 1):
            rows.append({"group": group, "rank": rank, "run_number": run_num,
                         "seed": BASE_SEED + run_num - 1, "distance": dist,
                         "trial_idx": trial_idx, "onset_s": onset_s,
                         "condition": condition})
    fieldnames = ["group", "rank", "run_number", "seed", "distance",
                  "trial_idx", "onset_s", "condition"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Long CSV saved to {out_path}  ({len(rows)} rows)")


def save_wide_csv(ranked_run_numbers, distances, top_n: int,
                  desired_freqs, seq_id_fn, out_path: Path) -> None:
    """One row per design — columns t01, t02, … show the condition sequence at a glance."""
    rows = []
    max_trials = 0
    all_data = list(_design_rows(
        ranked_run_numbers, distances, top_n, desired_freqs, seq_id_fn))
    max_trials = max(len(trials) for *_, trials in all_data)

    for group, rank, run_num, dist, trials in all_data:
        row = {"group": group, "rank": rank, "run_number": run_num,
               "seed": BASE_SEED + run_num - 1, "distance": dist}
        for i, (onset_s, condition) in enumerate(trials, 1):
            row[f"t{i:02d}"] = condition
        rows.append(row)

    trial_cols = [f"t{i:02d}" for i in range(1, max_trials + 1)]
    fieldnames = ["group", "rank", "run_number", "seed", "distance"] + trial_cols
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wide CSV saved to {out_path}  ({len(rows)} rows × {max_trials} trial columns)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank run designs by distance from mean predicted BOLD."
    )
    parser.add_argument("--output_dir", default="models_output/prf_notemporal")
    parser.add_argument("--active_cfs", nargs="+", type=int, default=ACTIVE_CFS,
                        help="CF indices to include (default: 7–18).")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of best designs to print and export (default: 20).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    npz_files  = sorted(output_dir.glob("*_notemporal_cf*_bold.npz"))

    all_dists   = []
    run_numbers = None

    for npz_path in npz_files:
        bold_matrix, rn, cf_index = load_bold_matrix(npz_path)
        if cf_index not in args.active_cfs:
            continue
        if run_numbers is None:
            run_numbers = rn
        all_dists.append(distance_from_mean(bold_matrix))
        print(f"  CF {cf_index:03d} | {bold_matrix.shape[0]} runs | "
              f"mean dist = {all_dists[-1].mean():.4f}")

    if not all_dists:
        print("No active CF files found.")
        return

    total_dist          = np.mean(all_dists, axis=0)
    ranked_indices      = np.argsort(total_dist)
    ranked_run_numbers  = np.array(run_numbers)[ranked_indices]
    ranked_distances    = total_dist[ranked_indices]

    print(f"\nRanked {len(total_dist)} designs across "
          f"{len(all_dists)} active CFs (indices {args.active_cfs}).\n")

    print(f"{'─'*55}")
    print(f"  Best {args.top_n} designs  (closest to mean)")
    print(f"{'─'*55}")
    for rank, (run_num, dist) in enumerate(
            zip(ranked_run_numbers[:args.top_n], ranked_distances[:args.top_n]), 1):
        print(f"  #{rank:>3}  run_{run_num:04d}  seed={BASE_SEED + run_num - 1:<6}  dist={dist:.4f}")

    print(f"\n{'─'*55}")
    print(f"  Worst {args.top_n} designs  (biggest outliers)")
    print(f"{'─'*55}")
    for rank, (run_num, dist) in enumerate(
            zip(ranked_run_numbers[-args.top_n:][::-1],
                ranked_distances[-args.top_n:][::-1]), 1):
        print(f"  #{rank:>3}  run_{run_num:04d}  seed={BASE_SEED + run_num - 1:<6}  dist={dist:.4f}")

    # save npz
    npz_path = output_dir / "design_ranking.npz"
    np.savez(npz_path,
             ranked_run_numbers=ranked_run_numbers,
             distances=ranked_distances,
             active_cfs=np.array(args.active_cfs))
    print(f"\nFull ranking saved to {npz_path}")

    # save CSVs with trial structure
    print("Reconstructing trial structures for CSV…")
    desired_freqs, seq_id_fn = _build_helpers()
    save_long_csv(ranked_run_numbers, ranked_distances, args.top_n,
                  desired_freqs, seq_id_fn, output_dir / "design_ranking_long.csv")
    save_wide_csv(ranked_run_numbers, ranked_distances, args.top_n,
                  desired_freqs, seq_id_fn, output_dir / "design_ranking_wide.csv")



if __name__ == "__main__":
    main()
