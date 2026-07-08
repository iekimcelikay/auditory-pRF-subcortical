"""tau_onoff_recovery_sanity_check.py
=====================================
2D parameter recovery check: can we identify separate ON and OFF AdapTrans
time constants from BOLD betas?

Strategy (linear decomposition — N+M passes instead of N×M):
    For each unique tau in TAU_GRID_MS, compute bold_on and bold_off
    separately, then combine in beta space:
        pred_betas(tau_on, tau_off) = rho * betas_on[tau_on] + betas_off[tau_off]
    For each GT pair: build noisy BOLD, fit betas, compute R² surface, save heatmap.
    Recovery succeeds if the R² peak is at (tau_on_gt, tau_off_gt).

Usage:
    python tau_onoff_recovery_sanity_check.py \\
        --results_dir models_output/toneclouds_gaussianprf_20260619_0114
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, convolve_hrf, SUBCORTICAL_PARAMS
from prf_models.pm_noise import PmNoise, apply_bold_noise

# ── Parameters ────────────────────────────────────────────────────────────────
# Ground truth pairs (tau_on_ms, tau_off_ms) — all values must be in TAU_GRID_MS
GT_PAIRS    = [(50, 50), (50, 200), (200, 50), (200, 200)]
TAU_GRID_MS = [25, 50, 100, 200, 400]

# Fixed (match tau_recovery_worker.py)
DESIGN_SEED = 42
W_VAL       = 0.8
RHO         = 1.0
CF_IDX      = 10
ALPHA       = 4.0
TR_S        = 1.6
TOTAL_RUN_DUR_S = 720.0
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
NOISE_SEED      = 42
NOISE_VOXEL     = "mid"
N_RUNS          = 24
SEQ_DUR_S       = 20.0
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"
SANITY_FREQ       = "fc572hz"

n_tr_per_run   = int(TOTAL_RUN_DUR_S / TR_S)
n_samp_per_run = int(TOTAL_RUN_DUR_S / SIGNAL_DT_S)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str,
                    default="models_output/toneclouds_gaussianprf_20260619_0114")
parser.add_argument("--out_dir", type=str, default="tau_onoff_recovery")
args = parser.parse_args()

results_dir = Path(args.results_dir)
out_dir     = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Phase 1: load cochlear PSTHs ──────────────────────────────────────────────
print("Loading cochlear PSTHs ...")
npz_files = sorted(results_dir.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {results_dir}/wav*/")

per_seq: dict = {}
cf_hz_used: float = None

for npz_path in npz_files:
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, CF_IDX)
    dt_s         = time_axis[1] - time_axis[0]
    total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
    sharpened    = apply_powerlaw_population(population_psth, ALPHA)[cf_index, :]
    if seq_id == TC_SILENCE_SEQ_ID:
        train = np.full(int(round(total_dur_ms)), float(np.mean(sharpened)))
    else:
        result, _, _ = chunk_from_id(sharpened, time_axis, seq_id)
        train = build_prf_boxcar_train(
            [np.mean(c) for c in result["chunks"]],
            result["onsets_ms"], result["offsets_ms"], total_dur_ms, dt_ms=1.0,
        )
    per_seq[seq_id] = {"train": train, "cf_hz": cf_hz}
    cf_hz_used = cf_hz

active_ids = sorted([k for k in per_seq if k != TC_SILENCE_SEQ_ID and SANITY_FREQ in k])
n_conds    = len(active_ids)
print(f"  CF={cf_hz_used:.0f} Hz | {n_conds} active sequences")

# ── HRF ──────────────────────────────────────────────────────────────────────
hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_run_design(seed: int) -> list:
    rng = np.random.RandomState(seed)
    onset_s = 10.0
    design: list = []
    while onset_s + SEQ_DUR_S <= TOTAL_RUN_DUR_S - 10.0:
        if rng.rand() < 0.25:
            design.append((TC_SILENCE_SEQ_ID, onset_s))
        else:
            design.append((rng.choice(active_ids), onset_s))
        onset_s += SEQ_DUR_S
    return design


def _build_glm_projector(run_designs: list) -> np.ndarray:
    n_tr_total = N_RUNS * n_tr_per_run
    X_int = np.zeros((n_tr_total, N_RUNS))
    for r in range(N_RUNS):
        X_int[r * n_tr_per_run:(r + 1) * n_tr_per_run, r] = 1.0
    X_cond = np.zeros((n_tr_total, n_conds))
    for r, rd in enumerate(run_designs):
        for seq_id, onset_s in rd:
            if seq_id not in active_ids:
                continue
            c      = active_ids.index(seq_id)
            boxcar = np.zeros(n_samp_per_run)
            i_on   = int(round(onset_s / SIGNAL_DT_S))
            i_off  = min(i_on + int(SEQ_DUR_S / SIGNAL_DT_S), n_samp_per_run)
            boxcar[i_on:i_off] = 1.0
            col = convolve_hrf(boxcar, hrf_kernel, signal_dt=SIGNAL_DT_S,
                               kernel_dt=SIGNAL_DT_S, output_dt=TR_S)
            X_cond[r * n_tr_per_run:(r + 1) * n_tr_per_run, c] += col[:n_tr_per_run]
    X_full = np.column_stack([X_int, X_cond])
    return np.linalg.pinv(X_full)


def fit_betas(bold_runs: list, projector: np.ndarray) -> np.ndarray:
    return (projector @ np.concatenate(bold_runs))[N_RUNS:]


def _r2_betas(predicted: np.ndarray, observed: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid  = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ── Build run designs and GLM projector ───────────────────────────────────────
run_designs = [_make_run_design(DESIGN_SEED + i) for i in range(N_RUNS)]
projector   = _build_glm_projector(run_designs)

# ── Forward pass: compute bold_on and bold_off for each unique tau ─────────────
# bold_combined(tau_on, tau_off) = rho * bold_on(tau_on) + bold_off(tau_off)
# Linearity of betas means: betas(tau_on, tau_off) = rho * betas_on[tau_on] + betas_off[tau_off]
# This reduces N^2 forward passes to 2N.
print(f"Computing forward passes for {len(TAU_GRID_MS)} tau values ×2 (ON + OFF) ...")
bold_on_runs:  dict[float, list] = {}
bold_off_runs: dict[float, list] = {}

for tau in TAU_GRID_MS:
    print(f"  tau={tau} ms ...")
    bold_on_runs[tau]  = []
    bold_off_runs[tau] = []
    for run_d in run_designs:
        result = assemble_run_bold(
            per_seq=per_seq, run_design=run_d,
            total_run_dur_s=TOTAL_RUN_DUR_S, hrf_kernel=hrf_kernel,
            cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
            w=W_VAL, K=None, apply_adaptrans_flag=True,
            rectify=RECTIFY, rho=RHO,
            tau_ms=tau, tau_ms_off=tau,
        )
        bold_on_runs[tau].append(result["bold_on"])
        bold_off_runs[tau].append(result["bold_off"])

# Beta-space representations (per tau, per channel)
betas_on  = {tau: fit_betas(bold_on_runs[tau],  projector) for tau in TAU_GRID_MS}
betas_off = {tau: fit_betas(bold_off_runs[tau], projector) for tau in TAU_GRID_MS}

# Precompute all (tau_on, tau_off) candidate predicted betas
pred_betas_grid: dict[tuple, np.ndarray] = {
    (t_on, t_off): RHO * betas_on[t_on] + betas_off[t_off]
    for t_on  in TAU_GRID_MS
    for t_off in TAU_GRID_MS
}

# ── Recovery check: per GT pair ───────────────────────────────────────────────
assert all(t in TAU_GRID_MS for pair in GT_PAIRS for t in pair), \
    "All GT_PAIRS values must appear in TAU_GRID_MS"

n_grid = len(TAU_GRID_MS)
n_pairs = len(GT_PAIRS)
fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), constrained_layout=True)
if n_pairs == 1:
    axes = [axes]

for ax, (tau_gt_on, tau_gt_off) in zip(axes, GT_PAIRS):
    # Build GT BOLD (from precomputed runs)
    gt_bold_combined = [
        RHO * on + off
        for on, off in zip(bold_on_runs[tau_gt_on], bold_off_runs[tau_gt_off])
    ]

    # Add noise and fit betas
    noisy_bold = [
        apply_bold_noise(b, PmNoise(voxel=NOISE_VOXEL, seed=NOISE_SEED + i), TR_S)
        for i, b in enumerate(gt_bold_combined)
    ]
    betas_noisy = fit_betas(noisy_bold, projector)

    # Compute R² surface
    r2_surf = np.zeros((n_grid, n_grid))
    for i, t_on in enumerate(TAU_GRID_MS):
        for j, t_off in enumerate(TAU_GRID_MS):
            r2_surf[i, j] = _r2_betas(pred_betas_grid[(t_on, t_off)], betas_noisy)

    best_flat = np.argmax(r2_surf)
    best_i, best_j = np.unravel_index(best_flat, r2_surf.shape)
    best_on  = TAU_GRID_MS[best_i]
    best_off = TAU_GRID_MS[best_j]
    recovered = (best_on == tau_gt_on) and (best_off == tau_gt_off)

    labels = [str(t) for t in TAU_GRID_MS]
    im = ax.imshow(r2_surf, origin="upper", aspect="auto",
                   vmin=r2_surf.min(), vmax=r2_surf.max(), cmap="viridis")
    ax.set_xticks(range(n_grid)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n_grid)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("tau_off (ms)")
    ax.set_ylabel("tau_on (ms)")
    ax.set_title(
        f"GT: on={tau_gt_on}, off={tau_gt_off}\n"
        f"Best: on={best_on}, off={best_off}  {'✓' if recovered else '✗'}",
        fontsize=9,
    )
    # mark ground truth
    gt_row = TAU_GRID_MS.index(tau_gt_on)
    gt_col = TAU_GRID_MS.index(tau_gt_off)
    ax.plot(gt_col, gt_row, "r+", ms=14, mew=2, label="GT")
    # mark best
    ax.plot(best_j, best_i, "w*", ms=10, label="Best")
    plt.colorbar(im, ax=ax, shrink=0.8, label="R²")

fig.suptitle(
    f"2D tau_on / tau_off recovery  |  CF={cf_hz_used:.0f} Hz  |  "
    f"w={W_VAL}  rho={RHO}  noise={NOISE_VOXEL}",
    fontsize=10,
)
out_path = out_dir / "tau_onoff_recovery_heatmap.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Recovery summary ─────────────────────────────────────────────")
for tau_gt_on, tau_gt_off in GT_PAIRS:
    gt_bold_combined = [
        RHO * on + off
        for on, off in zip(bold_on_runs[tau_gt_on], bold_off_runs[tau_gt_off])
    ]
    noisy_bold = [
        apply_bold_noise(b, PmNoise(voxel=NOISE_VOXEL, seed=NOISE_SEED + i), TR_S)
        for i, b in enumerate(gt_bold_combined)
    ]
    betas_noisy = fit_betas(noisy_bold, projector)
    r2_surf = np.array([
        [_r2_betas(pred_betas_grid[(t_on, t_off)], betas_noisy)
         for t_off in TAU_GRID_MS]
        for t_on in TAU_GRID_MS
    ])
    best_i, best_j = np.unravel_index(np.argmax(r2_surf), r2_surf.shape)
    best_on, best_off = TAU_GRID_MS[best_i], TAU_GRID_MS[best_j]
    recovered = (best_on == tau_gt_on) and (best_off == tau_gt_off)
    mark = "✓" if recovered else "✗"
    print(f"  {mark}  GT=({tau_gt_on:>3d},{tau_gt_off:>3d})  "
          f"→ best=({best_on:>3d},{best_off:>3d})  "
          f"R²={r2_surf[best_i, best_j]:.3f}")
