"""gt_multitau_fulltimecourse_singlerun_aggregate.py
======================================================
Reads all per-task .npz files from gt_multitau_fulltimecourse_singlerun_worker.py
(one per (CF, GT tau), each internally looping over 100 design seeds x 100
noise repeats) and produces a summary table plus, per (CF, GT tau): a
recovery-rate histogram and a full-timecourse R2-vs-tau curve.

Usage:
    python gt_multitau_fulltimecourse_singlerun_aggregate.py --results_dir gt_multitau_fulltimecourse_singlerun_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="gt_multitau_fulltimecourse_singlerun_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result files in {results_dir}")

tasks = []
for npz_path in npz_files:
    d = np.load(npz_path, allow_pickle=False)
    tau_grid = d["tau_grid_ms"].tolist()
    r2_mean = [np.mean(d[f"r2_tau{t}"]) for t in tau_grid]
    r2_std  = [np.std(d[f"r2_tau{t}"]) for t in tau_grid]
    tasks.append({
        "cf_idx": int(d["cf_idx"]), "cf_hz": float(d["cf_hz_used"]), "tau_gt": int(d["tau_gt"]),
        "bias": d["bias"], "recovery_rate": d["recovery_rate"],
        "tau_grid": tau_grid, "r2_mean": r2_mean, "r2_std": r2_std,
    })

tasks.sort(key=lambda t: (t["cf_idx"], t["tau_gt"]))
gt_taus = sorted({t["tau_gt"] for t in tasks})
cf_idxs = sorted({t["cf_idx"] for t in tasks})

print(f"\n{'─'*80}")
print(f"{'CF_IDX':>7}  {'CF(Hz)':>8}  {'tau_gt':>7}  {'bias':>9}  {'recov_mean':>10}")
print(f"{'─'*80}")
for t in tasks:
    print(f"{t['cf_idx']:>7}  {t['cf_hz']:>8.1f}  {t['tau_gt']:>7}  "
          f"{t['bias'].mean():>+9.2f}  {t['recovery_rate'].mean():>10.3f}")

# ── Plot: rows = GT tau, cols = CF ────────────────────────────────────────────
n_cf = len(cf_idxs)
n_gt = len(gt_taus)
fig, axes = plt.subplots(2 * n_gt, n_cf, figsize=(3.0 * n_cf, 3.4 * n_gt))
if n_cf == 1:
    axes = axes.reshape(-1, 1)

by_key = {(t["cf_idx"], t["tau_gt"]): t for t in tasks}
for row_gt_i, tau_gt in enumerate(gt_taus):
    for col_i, cf_idx in enumerate(cf_idxs):
        t = by_key.get((cf_idx, tau_gt))
        ax0 = axes[2 * row_gt_i, col_i]
        ax1 = axes[2 * row_gt_i + 1, col_i]
        if t is None:
            ax0.axis("off"); ax1.axis("off")
            continue
        ax0.hist(t["recovery_rate"], bins=15, color="steelblue", alpha=0.8)
        ax0.axvline(np.mean(t["recovery_rate"]), color="firebrick", linestyle="--")
        ax0.set_title(f"CF_IDX={cf_idx} ({t['cf_hz']:.0f}Hz)\nGT tau={tau_gt}ms", fontsize=7)
        ax0.tick_params(labelsize=6)

        ax1.errorbar(t["tau_grid"], t["r2_mean"], yerr=t["r2_std"], fmt="o-",
                     color="steelblue", capsize=2, markersize=3)
        ax1.axvline(tau_gt, color="firebrick", linestyle="--")
        ax1.tick_params(labelsize=6)
    axes[2 * row_gt_i, 0].set_ylabel(f"GT={tau_gt}ms\ncount", fontsize=7)
    axes[2 * row_gt_i + 1, 0].set_ylabel("pooled R2", fontsize=7)

fig.suptitle("Full-timecourse single-run recovery across multiple GT tau values, by CF", fontsize=10)
plt.tight_layout()
out_path = results_dir / "gt_multitau_fulltimecourse_singlerun_summary.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out_path.resolve()}")
