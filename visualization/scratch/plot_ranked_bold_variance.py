"""
plot_ranked_bold_variance.py
============================
Plot BOLD timeseries and variance for the top/bottom ranked run designs.

For each representative CF: shows the cross-design mean ± std envelope,
overlaid with the best-N and worst-N individual runs.

Usage
-----
    python visualization/scratch/plot_ranked_bold_variance.py
    python visualization/scratch/plot_ranked_bold_variance.py \
        --output_dir models_output/prf_notemporal_20260518_job5634143 \
        --cfs 10 16 24 --top_n 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_cf_matrix(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    cf = int(d["cf"])
    tr_s = float(d["tr_s"])
    run_keys = sorted(
        [k for k in d.files if k.startswith("run_")],
        key=lambda k: int(k.split("_")[1]),
    )
    run_numbers = [int(k.split("_")[1]) for k in run_keys]
    mat = np.stack([d[k] for k in run_keys], axis=0)  # (n_runs, n_trs)
    return mat, run_numbers, cf, tr_s


def load_ranking(output_dir: Path):
    d = np.load(output_dir / "design_ranking.npz", allow_pickle=True)
    return d["ranked_run_numbers"], d["distances"], d["active_cfs"]


def plot_cf_panel(ax, mat, run_numbers, ranked_run_numbers, distances,
                  top_n, tr_s, cf, show_worst=True):
    n_trs = mat.shape[1]
    t = np.arange(n_trs) * tr_s

    # cross-design mean and std
    mean_bold = mat.mean(axis=0)
    std_bold  = mat.std(axis=0)

    ax.fill_between(t, mean_bold - std_bold, mean_bold + std_bold,
                    alpha=0.15, color="gray", label="±1 SD (all runs)")
    ax.plot(t, mean_bold, color="black", lw=1.2, label="Mean (all runs)")

    run_num_to_idx = {rn: i for i, rn in enumerate(run_numbers)}

    best_runs  = ranked_run_numbers[:top_n]
    worst_runs = ranked_run_numbers[-top_n:][::-1]

    colors_best  = plt.cm.Blues(np.linspace(0.5, 0.9, top_n))
    colors_worst = plt.cm.Reds(np.linspace(0.5, 0.9, top_n))

    for rank, (run_num, color) in enumerate(zip(best_runs, colors_best), 1):
        if run_num not in run_num_to_idx:
            continue
        idx = run_num_to_idx[run_num]
        dist = distances[rank - 1]
        ax.plot(t, mat[idx], color=color, lw=0.8, alpha=0.8,
                label=f"Best #{rank} run_{run_num:04d} d={dist:.3f}")

    if show_worst:
        for rank, (run_num, color) in enumerate(zip(worst_runs, colors_worst), 1):
            if run_num not in run_num_to_idx:
                continue
            idx = run_num_to_idx[run_num]
            dist = distances[-(rank)]
            ax.plot(t, mat[idx], color=color, lw=0.8, alpha=0.8,
                    linestyle="--", label=f"Worst #{rank} run_{run_num:04d} d={dist:.3f}")

    ax.set_title(f"CF index {cf}", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("BOLD (a.u.)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.6)


def plot_distance_distribution(ax, distances, ranked_run_numbers, top_n):
    ax.hist(distances, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    for d in distances[:top_n]:
        ax.axvline(d, color="royalblue", lw=0.8, alpha=0.7)
    for d in distances[-top_n:]:
        ax.axvline(d, color="firebrick", lw=0.8, alpha=0.7)
    ax.set_xlabel("Distance from mean", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(f"Distance distribution — {len(distances)} designs", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axvline(distances[:top_n].mean(),  color="royalblue", lw=1.5, label=f"Best {top_n} mean")
    ax.axvline(distances[-top_n:].mean(), color="firebrick",  lw=1.5, label=f"Worst {top_n} mean")
    ax.legend(fontsize=7)


def main():
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models_output/prf_notemporal_20260518_job5634143")
    parser.add_argument("--cfs", nargs="+", type=int, default=None,
                        help="CF indices to plot. Defaults to all active CFs in the ranking.")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--show_worst", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ranked_run_numbers, distances, active_cfs = load_ranking(output_dir)

    cf_list = args.cfs if args.cfs is not None else list(active_cfs)

    out_path = output_dir / "ranked_bold_variance.pdf"
    with PdfPages(out_path) as pdf:
        for cf_idx in cf_list:
            npz_files = sorted(output_dir.glob(f"*_notemporal_cf{cf_idx:03d}_bold.npz"))
            if not npz_files:
                print(f"  No file for CF {cf_idx}, skipping.")
                continue
            mat, run_numbers, cf, tr_s = load_cf_matrix(npz_files[0])

            fig = plt.figure(figsize=(14, 8))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

            ax_ts   = fig.add_subplot(gs[0])
            ax_dist = fig.add_subplot(gs[1])

            plot_cf_panel(ax_ts, mat, run_numbers, ranked_run_numbers, distances,
                          args.top_n, tr_s, cf, show_worst=args.show_worst)
            plot_distance_distribution(ax_dist, distances, ranked_run_numbers, args.top_n)

            fig.suptitle(
                f"CF index {cf_idx} — best vs worst {args.top_n} designs  "
                f"(ranked across CFs {list(active_cfs)})",
                fontsize=10,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  CF {cf_idx} done")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
