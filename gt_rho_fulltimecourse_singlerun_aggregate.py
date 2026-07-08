"""gt_rho_fulltimecourse_singlerun_aggregate.py
=================================================
Reads all per-task .npz files from gt_rho_fulltimecourse_singlerun_worker.py
(one per CF, each internally looping over 100 design seeds x 100 noise
repeats) and produces a summary table plus, per CF: a full-timecourse R2 vs
tau curve and bias/recovery histograms for both tau and rho.

Usage:
    python gt_rho_fulltimecourse_singlerun_aggregate.py --results_dir gt_rho_fulltimecourse_singlerun_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="gt_rho_fulltimecourse_singlerun_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result files in {results_dir}")

tasks = []
for npz_path in npz_files:
    d = np.load(npz_path, allow_pickle=False)
    tau_grid = sorted(int(k.replace("r2_tau", "")) for k in d.files if k.startswith("r2_tau"))
    r2_mean = [np.mean(d[f"r2_tau{t}"]) for t in tau_grid]
    r2_std  = [np.std(d[f"r2_tau{t}"]) for t in tau_grid]
    tasks.append({
        "cf_idx": int(d["cf_idx"]), "cf_hz": float(d["cf_hz_used"]),
        "tau_gt": int(d["tau_gt"]), "rho_gt": float(d["rho_gt"]),
        "bias_tau": d["bias_tau"], "bias_rho": d["bias_rho"],
        "recovery_rate": d["recovery_rate"],
        "tau_grid": tau_grid, "r2_mean": r2_mean, "r2_std": r2_std,
    })

print(f"\n{'─'*90}")
print(f"{'CF_IDX':>7}  {'CF(Hz)':>8}  {'bias_tau':>9}  {'bias_rho':>9}  {'recov_mean':>10}")
print(f"{'─'*90}")
for t in tasks:
    print(f"{t['cf_idx']:>7}  {t['cf_hz']:>8.1f}  {t['bias_tau'].mean():>+9.2f}  "
          f"{t['bias_rho'].mean():>+9.3f}  {t['recovery_rate'].mean():>10.2f}")

n_cf = len(tasks)
fig, axes = plt.subplots(3, n_cf, figsize=(3.2 * n_cf, 9.5))
if n_cf == 1:
    axes = axes.reshape(3, 1)

for col_i, t in enumerate(tasks):
    ax0 = axes[0, col_i]
    ax0.hist(t["recovery_rate"], bins=15, color="steelblue", alpha=0.8)
    ax0.axvline(np.mean(t["recovery_rate"]), color="firebrick", linestyle="--")
    ax0.set_title(f"CF_IDX={t['cf_idx']} ({t['cf_hz']:.0f}Hz)", fontsize=8)
    ax0.set_xlabel("tau recovery rate", fontsize=7)
    ax0.tick_params(labelsize=6)

    ax1 = axes[1, col_i]
    ax1.errorbar(t["tau_grid"], t["r2_mean"], yerr=t["r2_std"], fmt="o-",
                 color="steelblue", capsize=3, markersize=4)
    ax1.axvline(t["tau_gt"], color="firebrick", linestyle="--")
    ax1.set_xlabel("tau candidate (ms)", fontsize=7)
    ax1.tick_params(labelsize=6)

    ax2 = axes[2, col_i]
    ax2.hist(t["bias_rho"], bins=15, color="darkorange", alpha=0.8)
    ax2.axvline(np.mean(t["bias_rho"]), color="firebrick", linestyle="--")
    ax2.axvline(0.0, color="black", linestyle=":")
    ax2.set_xlabel("rho bias", fontsize=7)
    ax2.tick_params(labelsize=6)

axes[0, 0].set_ylabel(f"count (of {len(tasks[0]['recovery_rate'])} designs)", fontsize=7)
axes[1, 0].set_ylabel("full-timecourse R2", fontsize=7)
axes[2, 0].set_ylabel("count (of designs)", fontsize=7)

fig.suptitle("Shared-tau grid search + linear-GLM rho recovery, full timecourse fit, by CF", fontsize=10)
plt.tight_layout()
out_path = results_dir / "gt_rho_fulltimecourse_singlerun_summary.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out_path.resolve()}")
