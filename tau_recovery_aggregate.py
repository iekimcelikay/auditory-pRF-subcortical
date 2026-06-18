"""tau_recovery_aggregate.py
============================
Reads all per-task .npz files from tau_recovery_worker.py and produces the
final summary plot (mean ± std R² across design seeds).

Usage:
    python tau_recovery_aggregate.py --results_dir tau_recovery_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DESIGN_SEEDS = [42, 137, 271, 500, 888]
W_SWEEP      = [0.3, 0.5, 0.8]
TAU_GT_SWEEP = [20, 75, 150, 250]
TAU_GRID_MS  = [20, 30, 50, 75, 100, 150, 250]

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="tau_recovery_results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
npz_files   = sorted(results_dir.glob("task_*.npz"))
print(f"Found {len(npz_files)} result files in {results_dir}")

n_expected = len(DESIGN_SEEDS) * len(W_SWEEP)
if len(npz_files) < n_expected:
    print(f"WARNING: expected {n_expected} files, got {len(npz_files)}. "
          f"Some tasks may have failed.")

# ── Collect ───────────────────────────────────────────────────────────────────
r2_noisy_seeds = {(w, tgt): [] for w in W_SWEEP for tgt in TAU_GT_SWEEP}
r2_nl_seeds    = {(w, tgt): [] for w in W_SWEEP for tgt in TAU_GT_SWEEP}
cf_hz_used     = None

for npz_path in npz_files:
    d     = np.load(npz_path, allow_pickle=False)
    w_val = float(d["w_val"])
    cf_hz_used = float(d["cf_hz_used"])

    for tau_gt in TAU_GT_SWEEP:
        r2_noisy_seeds[(w_val, tau_gt)].append(d[f"r2_noisy_{tau_gt}"].tolist())
        r2_nl_seeds   [(w_val, tau_gt)].append(d[f"r2_nl_{tau_gt}"].tolist())

# ── Aggregate ─────────────────────────────────────────────────────────────────
results = {}
for w_val in W_SWEEP:
    for tau_gt in TAU_GT_SWEEP:
        r2_n  = np.array(r2_noisy_seeds[(w_val, tau_gt)])
        r2_nl = np.array(r2_nl_seeds[(w_val, tau_gt)])
        results[(w_val, tau_gt)] = {
            "r2_noisy_mean": r2_n.mean(axis=0),
            "r2_noisy_std":  r2_n.std(axis=0),
            "r2_nl_mean":    r2_nl.mean(axis=0),
            "r2_nl_std":     r2_nl.std(axis=0),
            "n_seeds":       len(r2_n),
        }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"{'w':>5}  {'τ_GT':>6}  {'best_τ':>7}  {'ok':>4}  {'Δ R²(noisy)':>12}  seeds")
print(f"{'─'*65}")
for w_val in W_SWEEP:
    for tau_gt in TAU_GT_SWEEP:
        res   = results[(w_val, tau_gt)]
        best  = TAU_GRID_MS[int(np.argmax(res["r2_noisy_mean"]))]
        ok    = "✓" if best == tau_gt else f"✗→{best}"
        delta = max(res["r2_noisy_mean"]) - min(res["r2_noisy_mean"])
        print(f"  {w_val:>4}  {tau_gt:>6}  {best:>7}  {ok:>4}  {delta:>12.4f}  {res['n_seeds']}")

# ── Plot ───────────────────────────────────────────────────────────────────────
n_rows = len(TAU_GT_SWEEP)
n_cols = len(W_SWEEP)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
fig.suptitle(
    f"τ recovery — AdapTrans only | β-space R² | CF≈{cf_hz_used:.0f} Hz | "
    f"{len(DESIGN_SEEDS)} design seeds | noise='mid'",
    fontsize=10,
)

for col_i, w_val in enumerate(W_SWEEP):
    for row_i, tau_gt in enumerate(TAU_GT_SWEEP):
        ax  = axes[row_i, col_i]
        res = results[(w_val, tau_gt)]
        tg  = np.array(TAU_GRID_MS)

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

        ax.axvline(x=tau_gt, color="firebrick", linestyle="--",
                   linewidth=1.5, label=f"GT={tau_gt}ms")

        best = TAU_GRID_MS[int(np.argmax(res["r2_noisy_mean"]))]
        ok   = "✓" if best == tau_gt else f"✗→{best}ms"
        rng  = max(res["r2_noisy_mean"]) - min(res["r2_noisy_mean"])
        ax.set_title(f"w={w_val}  τ_GT={tau_gt}ms  {ok}  (Δ={rng:.4f})", fontsize=8)
        ax.set_xlabel("τ candidate (ms)", fontsize=8)
        ax.set_ylabel("R² (beta pattern)", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(TAU_GRID_MS)
        ax.set_xticklabels([str(t) for t in TAU_GRID_MS], fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=6)

plt.tight_layout()
out_path = Path("tau_recovery_w_sweep.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved → {out_path.resolve()}")
