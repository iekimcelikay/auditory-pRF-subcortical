"""
visualize_best_runs_bold.py
===========================
Plot predicted BOLD timecourses for the best-ranked run designs.

"Best" = closest to the cross-design mean BOLD (z-score MSE aggregated
across active CFs), matching the criterion in rank_run_designs.py.

Produces two figures:
  1. bold_best_runs_per_cf.png  — per-CF grid: top-N runs overlaid on mean
  2. bold_best_runs_summary.png — mean BOLD across best runs vs all runs,
                                   for each active CF

Usage
-----
    python visualization/scratch/visualize_best_runs_bold.py
    python visualization/scratch/visualize_best_runs_bold.py \\
        --output_dir models_output/prf_notemporal_20260515_job5623758 \\
        --active_cfs 7 8 9 10 11 --top_n 10 --save_dir figures/best_runs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

# ── defaults (match rank_run_designs.py / run_pipeline_notemporal.py) ─────────
ACTIVE_CFS = list(range(7, 19))
BASE_SEED  = 42


# ── data loading ──────────────────────────────────────────────────────────────

def load_bold_matrix(npz_path: Path) -> tuple:
    """Load bold_combined for all runs from one CF npz.

    Returns
    -------
    bold_matrix : np.ndarray, shape (n_runs, n_trs)
    run_numbers : list[int]   — 1-based run numbers
    tr_s        : float
    cf_index    : int
    """
    d = np.load(npz_path, allow_pickle=True)
    run_keys = sorted(
        (k for k in d.files if k.startswith("run_")),
        key=lambda k: int(k.split("_")[1]),
    )
    run_numbers = [int(k.split("_")[1]) for k in run_keys]
    bold_matrix = np.stack([d[k] for k in run_keys], axis=0)
    return bold_matrix, run_numbers, float(d["tr_s"]), int(d["cf"])


# ── ranking (same criterion as rank_run_designs.py) ───────────────────────────

def distance_from_mean(bold_matrix: np.ndarray) -> np.ndarray:
    """Mean squared z-score deviation of each run from the cross-design mean."""
    mean_bold = bold_matrix.mean(axis=0)
    std_bold  = bold_matrix.std(axis=0)
    std_bold  = np.where(std_bold > 0, std_bold, 1.0)
    z = (bold_matrix - mean_bold) / std_bold
    return (z ** 2).mean(axis=1)


def rank_runs(npz_files: list, active_cfs: list) -> tuple:
    """Aggregate z-score distance across active CFs and return ranked run numbers.

    Returns
    -------
    ranked_run_numbers : np.ndarray[int]   — 1-based, sorted best → worst
    ranked_distances   : np.ndarray[float]
    run_numbers        : list[int]         — original order (reference for indexing)
    tr_s               : float
    cf_data            : dict[int -> np.ndarray]  — bold matrices keyed by cf_index
    """
    all_dists  = []
    run_numbers = None
    tr_s        = 1.6
    cf_data     = {}

    for npz_path in npz_files:
        bold_matrix, rn, tr_s, cf_index = load_bold_matrix(npz_path)
        if cf_index not in active_cfs:
            continue
        if run_numbers is None:
            run_numbers = rn
        all_dists.append(distance_from_mean(bold_matrix))
        cf_data[cf_index] = bold_matrix
        print(f"  CF {cf_index:03d} | {bold_matrix.shape[0]} runs × "
              f"{bold_matrix.shape[1]} TRs | mean dist = {all_dists[-1].mean():.4f}")

    if not all_dists:
        raise ValueError("No active CF files found.")

    total_dist         = np.mean(all_dists, axis=0)
    ranked_indices     = np.argsort(total_dist)
    ranked_run_numbers = np.array(run_numbers)[ranked_indices]
    ranked_distances   = total_dist[ranked_indices]

    return ranked_run_numbers, ranked_distances, run_numbers, tr_s, cf_data


# ── figure 1: per-CF, overlay top-N BOLD timecourses ─────────────────────────

def plot_best_runs_per_cf(
        cf_data: dict,
        run_numbers: list,
        ranked_run_numbers: np.ndarray,
        ranked_distances: np.ndarray,
        tr_s: float,
        top_n: int,
        save_path: Path,
) -> None:
    """Grid of CF rows; each row overlays top-N runs on the ensemble mean."""
    cf_indices = sorted(cf_data.keys())
    n_cfs = len(cf_indices)
    fig, axes = plt.subplots(n_cfs, 1, figsize=(15, 3.2 * n_cfs), sharex=True, squeeze=False)

    # colour gradient: best = dark blue, N-th best = light blue
    colours = cm.Blues(np.linspace(0.85, 0.35, top_n))

    run_num_to_pos = {rn: i for i, rn in enumerate(run_numbers)}
    best_run_nums  = ranked_run_numbers[:top_n]
    best_distances = ranked_distances[:top_n]

    for row, cf_index in enumerate(cf_indices):
        ax = axes[row, 0]
        bold_matrix = cf_data[cf_index]
        n_trs = bold_matrix.shape[1]
        t_s   = np.arange(n_trs) * tr_s

        # ensemble mean across *all* runs as grey background reference
        all_mean = bold_matrix.mean(axis=0)
        ax.fill_between(
            t_s,
            np.percentile(bold_matrix, 5, axis=0),
            np.percentile(bold_matrix, 95, axis=0),
            alpha=0.12, color="grey", label="all runs 5–95 %",
        )
        ax.plot(t_s, all_mean, color="grey", lw=1.0, ls="--",
                alpha=0.7, label="all-run mean", zorder=2)

        # top-N best runs, darkest = rank 1
        for rank_idx, (run_num, dist) in enumerate(zip(best_run_nums, best_distances)):
            pos   = run_num_to_pos[run_num]
            bold  = bold_matrix[pos]
            lw    = 1.8 if rank_idx == 0 else 0.9
            zord  = top_n + 3 - rank_idx
            label = f"run {run_num:04d} (dist={dist:.3f})" if rank_idx < 3 else None
            ax.plot(t_s, bold, color=colours[rank_idx], lw=lw, zorder=zord,
                    label=label, alpha=0.9)

        ax.set_ylabel("BOLD (a.u.)", fontsize=9)
        ax.set_title(f"CF index {cf_index}  —  top {top_n} best runs overlaid",
                     fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7,
                      ncol=2, handlelength=1.5)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        f"Best {top_n} run designs — predicted BOLD timecourses\n"
        "Ranked by mean z-score distance from cross-design average (aggregated across active CFs)\n"
        "Dark blue = rank 1 (most representative);  grey band = all-run 5–95 %",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── figure 2: best-run mean vs all-run mean per CF ────────────────────────────

def plot_best_vs_all_mean(
        cf_data: dict,
        run_numbers: list,
        ranked_run_numbers: np.ndarray,
        tr_s: float,
        top_n: int,
        save_path: Path,
) -> None:
    """Compare mean BOLD of top-N best runs vs the full ensemble mean."""
    cf_indices = sorted(cf_data.keys())
    n_cfs = len(cf_indices)
    fig, axes = plt.subplots(n_cfs, 1, figsize=(15, 2.8 * n_cfs), sharex=True, squeeze=False)

    run_num_to_pos = {rn: i for i, rn in enumerate(run_numbers)}
    best_positions = [run_num_to_pos[rn] for rn in ranked_run_numbers[:top_n]]

    for row, cf_index in enumerate(cf_indices):
        ax = axes[row, 0]
        bold_matrix = cf_data[cf_index]
        n_trs = bold_matrix.shape[1]
        t_s   = np.arange(n_trs) * tr_s

        all_mean  = bold_matrix.mean(axis=0)
        best_mean = bold_matrix[best_positions].mean(axis=0)
        best_std  = bold_matrix[best_positions].std(axis=0)

        ax.fill_between(t_s, best_mean - best_std, best_mean + best_std,
                        alpha=0.25, color="steelblue")
        ax.plot(t_s, all_mean,  color="grey",      lw=1.2, ls="--", label="all-run mean")
        ax.plot(t_s, best_mean, color="steelblue", lw=1.6,          label=f"top-{top_n} mean ± σ")

        ax.set_ylabel("BOLD (a.u.)", fontsize=9)
        ax.set_title(f"CF index {cf_index}", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        f"Top-{top_n} best runs vs full ensemble — mean predicted BOLD per CF",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize predicted BOLD for the best-ranked run designs."
    )
    parser.add_argument("--output_dir",
                        default="models_output/prf_notemporal_20260515_job5623758",
                        help="Directory containing *_notemporal_cf*_bold.npz files.")
    parser.add_argument("--save_dir", default=None,
                        help="Where to save figures (default: same as output_dir).")
    parser.add_argument("--active_cfs", nargs="+", type=int, default=ACTIVE_CFS,
                        help=f"CF indices to use for ranking (default: {ACTIVE_CFS}).")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of best runs to visualize (default: 10).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir   = Path(args.save_dir) if args.save_dir else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(output_dir.glob("*_notemporal_cf*_bold.npz"))
    if not npz_files:
        print(f"No matching npz files in {output_dir}")
        return
    print(f"Found {len(npz_files)} CF file(s). Ranking across active CFs {args.active_cfs}…")

    ranked_run_numbers, ranked_distances, run_numbers, tr_s, cf_data = rank_runs(
        npz_files, args.active_cfs
    )

    print(f"\nTop {args.top_n} runs (best → most representative design):")
    for rank, (rn, dist) in enumerate(
            zip(ranked_run_numbers[:args.top_n], ranked_distances[:args.top_n]), 1):
        print(f"  #{rank:>3}  run_{rn:04d}  seed={BASE_SEED + rn - 1:<6}  dist={dist:.4f}")

    plot_best_runs_per_cf(
        cf_data, run_numbers, ranked_run_numbers, ranked_distances, tr_s,
        top_n    = args.top_n,
        save_path= save_dir / "bold_best_runs_per_cf.png",
    )
    plot_best_vs_all_mean(
        cf_data, run_numbers, ranked_run_numbers, tr_s,
        top_n    = args.top_n,
        save_path= save_dir / "bold_best_runs_summary.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
