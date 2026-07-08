"""tau_sus_recovery_worker.py
==============================
Single worker for the sustained-channel time-constant (tau_sus) recovery sweep.
Called by the SLURM array job — each task handles one (design_seed, beta_sus) pair.

Task mapping (20 tasks = 5 seeds × 4 beta_sus values):
    task_id = ds_idx * len(BETA_SUS_SWEEP) + beta_sus_idx

All TAU_SUS_GT_SWEEP values are handled within each task by precomputing
forward runs for every TAU_SUS_GRID_MS point (GT values are a subset of the grid).
"""

import argparse
import numpy as np
from pathlib import Path

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, convolve_hrf, SUBCORTICAL_PARAMS
from prf_models.pm_noise import PmNoise, apply_bold_noise

# ── Sweep parameters (must match tau_sus_recovery_aggregate.py) ───────────────
DESIGN_SEEDS     = [42, 137, 271, 500, 888]
BETA_SUS_SWEEP   = [0.0, 0.5, 1.0, 2.0]
TAU_SUS_GT_SWEEP = [50, 100, 200, 400]    # must all be in TAU_SUS_GRID_MS
TAU_SUS_GRID_MS  = [25, 50, 100, 200, 400, 800]

# ── Fixed parameters ───────────────────────────────────────────────────────────
RESULTS_DIR     = Path("models_output/toneclouds_gaussianprf_20260619_0114")
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"
CF_IDX          = 12      # 549 Hz — nearest to 572 Hz tone-cloud band
ALPHA           = 4.0
W               = 0.8
RHO             = 1.0
TAU_MS          = 100.0   # AdapTrans time constant (fixed)
TR_S            = 1.6
TOTAL_RUN_DUR_S = 720.0
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
NOISE_SEED      = 42
NOISE_VOXEL     = "mid"
N_RUNS          = 24
SEQ_DUR_S       = 20.0

n_tr_per_run   = int(TOTAL_RUN_DUR_S / TR_S)
n_samp_per_run = int(TOTAL_RUN_DUR_S / SIGNAL_DT_S)

assert all(gt in TAU_SUS_GRID_MS for gt in TAU_SUS_GT_SWEEP), \
    "All GT values must be in the grid so forward runs can be reused."

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True)
parser.add_argument("--out_dir", type=str, default="tau_sus_recovery_results")
args = parser.parse_args()

ds_idx       = args.task_id // len(BETA_SUS_SWEEP)
beta_sus_idx = args.task_id % len(BETA_SUS_SWEEP)
ds_seed      = DESIGN_SEEDS[ds_idx]
beta_sus_val = BETA_SUS_SWEEP[beta_sus_idx]

out_dir  = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_ds{ds_seed}_bsus{beta_sus_val}.npz"

print(f"Task {args.task_id}: design_seed={ds_seed}, beta_sus={beta_sus_val}")
print(f"Output → {out_path}")

# ── Phase 1: load cochlear PSTHs ──────────────────────────────────────────────
print(f"Loading cochlear PSTHs (CF index {CF_IDX}) ...")
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {RESULTS_DIR}/wav*/")

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

active_ids = sorted(k for k in per_seq if k != TC_SILENCE_SEQ_ID)
print(f"  CF={cf_hz_used:.0f} Hz | {len(active_ids)} active sequences")

# ── HRF kernel ─────────────────────────────────────────────────────────────────
hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)


# ── Run design ────────────────────────────────────────────────────────────────
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


run_designs = [_make_run_design(ds_seed + i) for i in range(N_RUNS)]


# ── GLM projector (beta-space) ────────────────────────────────────────────────
def _build_glm_projector() -> np.ndarray:
    n_tr_total = N_RUNS * n_tr_per_run
    n_conds    = len(active_ids)
    X_int  = np.zeros((n_tr_total, N_RUNS))
    X_cond = np.zeros((n_tr_total, n_conds))
    for r in range(N_RUNS):
        X_int[r * n_tr_per_run:(r + 1) * n_tr_per_run, r] = 1.0
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
    return np.linalg.pinv(np.column_stack([X_int, X_cond]))


def _fit_betas(bold_runs: list, projector: np.ndarray) -> np.ndarray:
    return (projector @ np.concatenate(bold_runs))[N_RUNS:]


def _r2_betas(predicted: np.ndarray, observed: np.ndarray) -> float:
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    if ss_tot == 0:
        return np.nan
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid  = observed - X @ beta
    return 1.0 - float(np.dot(resid, resid)) / ss_tot


print(f"Building GLM projector ({N_RUNS} runs, {len(active_ids)} conditions) ...")
projector = _build_glm_projector()


# ── Forward model ─────────────────────────────────────────────────────────────
def _forward_run(tau_sus_ms_val: float, run_design: list) -> np.ndarray:
    return assemble_run_bold(
        per_seq=per_seq,
        run_design=run_design,
        total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel,
        cf_hz=cf_hz_used,
        tr_s=TR_S,
        signal_dt_s=SIGNAL_DT_S,
        w=W, K=None,
        apply_adaptrans_flag=True,
        rectify=RECTIFY,
        rho=RHO,
        tau_ms=TAU_MS,
        tau_sus_ms=tau_sus_ms_val,
        beta_sus=beta_sus_val,
    )["bold_combined"]


# ── Compute: precompute all grid runs, reuse GT ───────────────────────────────
print(f"Computing {len(TAU_SUS_GRID_MS)} × {N_RUNS} forward runs ...")
bold_runs  = {t: [_forward_run(t, d) for d in run_designs] for t in TAU_SUS_GRID_MS}
betas_grid = {t: _fit_betas(bold_runs[t], projector) for t in TAU_SUS_GRID_MS}

r2_noisy_all = {}
r2_nl_all    = {}

for tau_sus_gt in TAU_SUS_GT_SWEEP:
    gt_runs    = bold_runs[tau_sus_gt]
    noise_base = NOISE_SEED + ds_idx * N_RUNS
    noisy_runs = [
        apply_bold_noise(bg, PmNoise(voxel=NOISE_VOXEL, seed=noise_base + i), TR_S)
        for i, bg in enumerate(gt_runs)
    ]
    betas_noisy = _fit_betas(noisy_runs, projector)
    betas_gt    = betas_grid[tau_sus_gt]

    r2_noisy_all[tau_sus_gt] = [_r2_betas(betas_grid[t], betas_noisy) for t in TAU_SUS_GRID_MS]
    r2_nl_all[tau_sus_gt]    = [_r2_betas(betas_grid[t], betas_gt)    for t in TAU_SUS_GRID_MS]

    best = TAU_SUS_GRID_MS[int(np.nanargmax(r2_noisy_all[tau_sus_gt]))]
    ok   = "✓" if best == tau_sus_gt else "✗"
    print(f"  τ_sus_GT={tau_sus_gt}ms | best={best}ms {ok}")

# ── Save ──────────────────────────────────────────────────────────────────────
save_dict = {
    "task_id":        args.task_id,
    "ds_idx":         ds_idx,
    "design_seed":    ds_seed,
    "beta_sus_val":   beta_sus_val,
    "cf_hz_used":     cf_hz_used,
    "TAU_SUS_GRID_MS": TAU_SUS_GRID_MS,
    "TAU_SUS_GT_SWEEP": TAU_SUS_GT_SWEEP,
}
for tau_sus_gt in TAU_SUS_GT_SWEEP:
    save_dict[f"r2_noisy_{tau_sus_gt}"] = np.array(r2_noisy_all[tau_sus_gt])
    save_dict[f"r2_nl_{tau_sus_gt}"]    = np.array(r2_nl_all[tau_sus_gt])

np.savez(out_path, **save_dict)
print(f"Saved → {out_path}")
