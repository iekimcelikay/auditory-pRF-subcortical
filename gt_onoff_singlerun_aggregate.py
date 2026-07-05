"""gt_onoff_singlerun_aggregate.py
==================================
Reads all per-task .npz files from gt_onoff_singlerun_worker.py (one per
(CF, GT pair), each internally looping over 100 design seeds x 100 noise
repeats) and produces a summary table plus, per (CF, GT pair): a pooled
tau_on x tau_off R2 heatmap and a recovery-rate histogram.

Usage:
    python gt_onoff_singlerun_aggregate.py --results_dir gt_onoff_singlerun_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="gt_onoff_singlerun_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result files in {results_dir}")

tasks = []
for npz_path in npz_files:
    d = np.load(npz_path, allow_pickle=False)
    tau_grid = d["tau_grid_ms"].tolist()
    r2_mean = np.array([[np.mean(d[f"r2_ton{ton}_toff{toff}"]) for toff in tau_grid] for ton in tau_grid])
    tasks.append({
        "task_id": int(d["task_id"]), "cf_idx": int(d["cf_idx"]), "cf_hz": float(d["cf_hz_used"]),
        "tau_on_gt": int(d["tau_on_gt"]), "tau_off_gt": int(d["tau_off_gt"]),
        "bias_on": d["bias_on"], "bias_off": d["bias_off"], "recovery_rate": d["recovery_rate"],
        "tau_grid": tau_grid, "r2_mean": r2_mean,
    })

print(f"\n{'─'*90}")
print(f"{'CF_IDX':>7}  {'CF(Hz)':>8}  {'GT(on,off)':>12}  {'bias_on':>9}  {'bias_off':>9}  {'recov_mean':>10}")
print(f"{'─'*90}")
for t in tasks:
    print(f"{t['cf_idx']:>7}  {t['cf_hz']:>8.1f}  "
          f"{'(' + str(t['tau_on_gt']) + ',' + str(t['tau_off_gt']) + ')':>12}  "
          f"{t['bias_on'].mean():>+9.2f}  {t['bias_off'].mean():>+9.2f}  {t['recovery_rate'].mean():>10.2f}")

# ── Plot: per-task recovery-rate histogram + pooled R2 heatmap ───────────────
n_tasks = len(tasks)
fig, axes = plt.subplots(2, n_tasks, figsize=(3.4 * n_tasks, 7.0))
if n_tasks == 1:
    axes = axes.reshape(2, 1)

for col_i, t in enumerate(tasks):
    ax = axes[0, col_i]
    ax.hist(t["recovery_rate"], bins=15, color="steelblue", alpha=0.8)
    ax.axvline(np.mean(t["recovery_rate"]), color="firebrick", linestyle="--")
    ax.set_title(f"CF_IDX={t['cf_idx']} ({t['cf_hz']:.0f}Hz)\nGT=({t['tau_on_gt']},{t['tau_off_gt']})", fontsize=8)
    ax.set_xlabel("recovery rate", fontsize=7)
    ax.tick_params(labelsize=6)

    ax2 = axes[1, col_i]
    im = ax2.imshow(t["r2_mean"], origin="lower", cmap="viridis", aspect="auto")
    ax2.set_xticks(range(len(t["tau_grid"])))
    ax2.set_xticklabels(t["tau_grid"], fontsize=6)
    ax2.set_yticks(range(len(t["tau_grid"])))
    ax2.set_yticklabels(t["tau_grid"], fontsize=6)
    gt_x = t["tau_grid"].index(t["tau_off_gt"])
    gt_y = t["tau_grid"].index(t["tau_on_gt"])
    ax2.plot(gt_x, gt_y, marker="x", color="red", markersize=10, markeredgewidth=2)
    ax2.set_xlabel("tau_off candidate (ms)", fontsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

axes[0, 0].set_ylabel(f"count (of {len(tasks[0]['recovery_rate'])} designs)", fontsize=7)
axes[1, 0].set_ylabel("tau_on candidate (ms)", fontsize=7)

fig.suptitle("ON/OFF tau recovery: pooled R2 heatmap (red x = GT) + recovery-rate histogram, by (CF, GT)", fontsize=10)
plt.tight_layout()
out_path = results_dir / "gt_onoff_singlerun_summary.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out_path.resolve()}")
