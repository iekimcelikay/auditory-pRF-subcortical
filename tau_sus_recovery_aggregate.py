"""tau_sus_recovery_aggregate.py
=================================
Reads all per-task .npz files from tau_sus_recovery_worker.py and produces
a summary plot: R² curves (mean ± std across design seeds) for each
(beta_sus, tau_sus_gt) combination.

Usage:
    python tau_sus_recovery_aggregate.py --results_dir tau_sus_recovery_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DESIGN_SEEDS     = [42, 137, 271, 500, 888]
BETA_SUS_SWEEP   = [0.0, 0.5, 1.0, 2.0]
TAU_SUS_GT_SWEEP = [50, 100, 200, 400]
TAU_SUS_GRID_MS  = [25, 50, 100, 200, 400, 800]

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="tau_sus_recovery_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files   = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result file(s) in {results_dir}")

n_expected = len(DESIGN_SEEDS) * len(BETA_SUS_SWEEP)
if len(npz_files) < n_expected:
    print(f"WARNING: expected {n_expected} files, got {len(npz_files)}.")

# ── Collect ───────────────────────────────────────────────────────────────────
r2_noisy_seeds = {(b, t): [] for b in BETA_SUS_SWEEP for t in TAU_SUS_GT_SWEEP}
r2_nl_seeds    = {(b, t): [] for b in BETA_SUS_SWEEP for t in TAU_SUS_GT_SWEEP}
cf_hz_used     = None

for npz_path in npz_files:
    d            = np.load(npz_path, allow_pickle=False)
    beta_sus_val = float(d["beta_sus_val"])
    cf_hz_used   = float(d["cf_hz_used"])

    for tau_sus_gt in TAU_SUS_GT_SWEEP:
        r2_noisy_seeds[(beta_sus_val, tau_sus_gt)].append(d[f"r2_noisy_{tau_sus_gt}"].tolist())
        r2_nl_seeds   [(beta_sus_val, tau_sus_gt)].append(d[f"r2_nl_{tau_sus_gt}"].tolist())

# ── Aggregate ─────────────────────────────────────────────────────────────────
results = {}
for beta_sus_val in BETA_SUS_SWEEP:
    for tau_sus_gt in TAU_SUS_GT_SWEEP:
        r2_n  = np.array(r2_noisy_seeds[(beta_sus_val, tau_sus_gt)])
        r2_nl = np.array(r2_nl_seeds[(beta_sus_val, tau_sus_gt)])
        results[(beta_sus_val, tau_sus_gt)] = {
            "r2_noisy_mean": np.nanmean(r2_n,  axis=0),
            "r2_noisy_std":  np.nanstd(r2_n,   axis=0),
            "r2_nl_mean":    np.nanmean(r2_nl,  axis=0),
            "r2_nl_std":     np.nanstd(r2_nl,   axis=0),
            "n_seeds":       len(r2_n),
        }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"{'beta_sus':>9}  {'τ_sus_GT':>9}  {'best_τ':>7}  {'ok':>6}  {'ΔR²(noisy)':>11}  seeds")
print(f"{'─'*70}")
for beta_sus_val in BETA_SUS_SWEEP:
    for tau_sus_gt in TAU_SUS_GT_SWEEP:
        res   = results[(beta_sus_val, tau_sus_gt)]
        best  = TAU_SUS_GRID_MS[int(np.nanargmax(res["r2_noisy_mean"]))]
        ok    = "✓" if best == tau_sus_gt else f"✗→{best}"
        delta = float(np.nanmax(res["r2_noisy_mean"]) - np.nanmin(res["r2_noisy_mean"]))
        print(f"  {beta_sus_val:>7}  {tau_sus_gt:>9}  {best:>7}  {ok:>6}  {delta:>11.4f}  {res['n_seeds']}")

# ── Plot: rows = tau_sus_gt, cols = beta_sus ───────────────────────────────────
n_rows = len(TAU_SUS_GT_SWEEP)
n_cols = len(BETA_SUS_SWEEP)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                          sharex=True, sharey=True)
fig.suptitle(
    f"τ_sus recovery (beta-space R²) | CF≈{cf_hz_used:.0f} Hz | "
    f"{len(DESIGN_SEEDS)} seeds | noise='mid'",
    fontsize=11,
)

tg = np.array(TAU_SUS_GRID_MS)

for row_i, tau_sus_gt in enumerate(TAU_SUS_GT_SWEEP):
    for col_i, beta_sus_val in enumerate(BETA_SUS_SWEEP):
        ax  = axes[row_i, col_i]
        res = results[(beta_sus_val, tau_sus_gt)]

        ax.plot(tg, res["r2_nl_mean"], "o--", color="forestgreen",
                linewidth=1.5, markersize=5, alpha=0.8, label="noiseless")
        ax.fill_between(tg,
                         res["r2_nl_mean"] - res["r2_nl_std"],
                         res["r2_nl_mean"] + res["r2_nl_std"],
                         color="forestgreen", alpha=0.15)

        ax.plot(tg, res["r2_noisy_mean"], "o-", color="steelblue",
                linewidth=2, markersize=7, label=f"noisy (n={res['n_seeds']})")
        ax.fill_between(tg,
                         res["r2_noisy_mean"] - res["r2_noisy_std"],
                         res["r2_noisy_mean"] + res["r2_noisy_std"],
                         color="steelblue", alpha=0.2)

        ax.axvline(x=tau_sus_gt, color="firebrick", linestyle="--",
                   linewidth=1.5, label=f"GT={tau_sus_gt}ms")

        best = TAU_SUS_GRID_MS[int(np.nanargmax(res["r2_noisy_mean"]))]
        ok   = "✓" if best == tau_sus_gt else f"✗→{best}ms"
        rng  = float(np.nanmax(res["r2_noisy_mean"]) - np.nanmin(res["r2_noisy_mean"]))
        ax.set_title(f"β_sus={beta_sus_val}  τ_GT={tau_sus_gt}ms  {ok}  (Δ={rng:.4f})",
                     fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(TAU_SUS_GRID_MS)
        ax.set_xticklabels([str(t) for t in TAU_SUS_GRID_MS], fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        if row_i == n_rows - 1:
            ax.set_xlabel("τ_sus candidate (ms)", fontsize=8)
        if col_i == 0:
            ax.set_ylabel("R² (beta space)", fontsize=8)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=6)

plt.tight_layout()
out_path = results_dir / "tau_sus_recovery_w_sweep.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved → {out_path.resolve()}")
