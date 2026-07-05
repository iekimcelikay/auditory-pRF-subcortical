"""gt35_singlerun_aggregate.py
==============================
Reads all per-task .npz files from gt35_singlerun_worker.py (one per
(design_seed, CF) pair) and produces a summary: bias/recovery-rate
distribution across the different trial orderings, broken down by CF, plus a
combined R2-vs-tau-candidate plot pooling all design seeds per CF.

Usage:
    python gt35_singlerun_aggregate.py --results_dir gt35_singlerun_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TAU_GRID_MS = [20, 35, 45, 50, 75]
TAU_GT = 35
CF_INDICES = [0, 6, 10, 12, 14, 17, 20, 21, 25, 29]

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="gt35_singlerun_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result files in {results_dir}")

by_cf = {cf: {"bias": [], "std": [], "recovery_rate": [], "design_seed": [],
              "r2_pooled": {tau: [] for tau in TAU_GRID_MS}, "cf_hz": None}
         for cf in CF_INDICES}

for npz_path in npz_files:
    d = np.load(npz_path, allow_pickle=False)
    cf_idx = int(d["cf_idx"])
    b = by_cf[cf_idx]
    b["bias"].extend(d["bias"].tolist())
    b["std"].extend(d["std"].tolist())
    b["recovery_rate"].extend(d["recovery_rate"].tolist())
    b["design_seed"].extend(d["design_seeds"].tolist())
    b["cf_hz"] = float(d["cf_hz_used"])
    for tau in TAU_GRID_MS:
        b["r2_pooled"][tau].extend(d[f"r2_tau{tau}"].tolist())

print(f"\n{'─'*80}")
print(f"{'CF_IDX':>7}  {'CF(Hz)':>8}  {'n_designs':>9}  {'bias_mean':>10}  "
      f"{'recov_mean':>10}  {'recov_range':>13}")
print(f"{'─'*80}")
for cf_idx in CF_INDICES:
    b = by_cf[cf_idx]
    n = len(b["bias"])
    if n == 0:
        print(f"{cf_idx:>7}  {'--':>8}  {0:>9}  {'--':>10}  {'--':>10}  {'--':>13}")
        continue
    bias_arr = np.array(b["bias"])
    recov_arr = np.array(b["recovery_rate"])
    print(f"{cf_idx:>7}  {b['cf_hz']:>8.1f}  {n:>9}  {bias_arr.mean():>+10.2f}  "
          f"{recov_arr.mean():>10.2f}  [{recov_arr.min():.2f}, {recov_arr.max():.2f}]")

# ── Overall best/worst (design, CF) combo ─────────────────────────────────────
all_combos = [(cf_idx, i, by_cf[cf_idx]["recovery_rate"][i], by_cf[cf_idx]["bias"][i],
               by_cf[cf_idx]["design_seed"][i])
              for cf_idx in CF_INDICES for i in range(len(by_cf[cf_idx]["recovery_rate"]))]
if all_combos:
    best = max(all_combos, key=lambda x: x[2])
    worst = min(all_combos, key=lambda x: x[2])
    print(f"\nBest combo:  CF_IDX={best[0]}  seed={best[4]}  recovery={best[2]:.2f}  bias={best[3]:+.2f}ms")
    print(f"Worst combo: CF_IDX={worst[0]}  seed={worst[4]}  recovery={worst[2]:.2f}  bias={worst[3]:+.2f}ms")

# ── Plot: recovery-rate distribution and pooled R2 curve, per CF ─────────────
n_cf = len(CF_INDICES)
fig, axes = plt.subplots(2, n_cf, figsize=(3.2 * n_cf, 6.5))
for col_i, cf_idx in enumerate(CF_INDICES):
    b = by_cf[cf_idx]
    ax = axes[0, col_i]
    if b["recovery_rate"]:
        ax.hist(b["recovery_rate"], bins=15, color="steelblue", alpha=0.8)
        ax.axvline(np.mean(b["recovery_rate"]), color="firebrick", linestyle="--")
    ax.set_title(f"CF_IDX={cf_idx}\n({b['cf_hz']:.0f}Hz)" if b["cf_hz"] else f"CF_IDX={cf_idx}", fontsize=8)
    ax.set_xlabel("recovery rate", fontsize=7)
    ax.tick_params(labelsize=6)

    ax2 = axes[1, col_i]
    if b["recovery_rate"]:
        means = [np.mean(b["r2_pooled"][t]) for t in TAU_GRID_MS]
        stds = [np.std(b["r2_pooled"][t]) for t in TAU_GRID_MS]
        ax2.errorbar(TAU_GRID_MS, means, yerr=stds, fmt="o-", color="steelblue",
                     capsize=3, markersize=4)
        ax2.axvline(TAU_GT, color="firebrick", linestyle="--")
    ax2.set_xlabel("tau candidate (ms)", fontsize=7)
    ax2.tick_params(labelsize=6)
axes[0, 0].set_ylabel(f"count (of {len(by_cf[CF_INDICES[0]]['bias'])} designs)", fontsize=7)
axes[1, 0].set_ylabel("pooled R2", fontsize=7)

fig.suptitle(f"GT tau={TAU_GT}ms | single-run recovery across designs, by CF", fontsize=10)
plt.tight_layout()
out_path = results_dir / "gt35_singlerun_by_cf.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out_path.resolve()}")
