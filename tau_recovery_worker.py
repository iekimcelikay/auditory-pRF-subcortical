"""tau_recovery_worker.py
========================
Single worker for the tau recovery sweep.
Called by the SLURM array job — each task handles one (design_seed, w) pair.

Usage (by SLURM):
    python tau_recovery_worker.py --task_id <0..14> --out_dir <dir>

Task mapping (15 tasks = 5 seeds × 3 w values):
    task_id = ds_idx * len(W_SWEEP) + w_idx
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

# ── Sweep parameters (must match tau_recovery_aggregate.py) ───────────────────
DESIGN_SEEDS  = [42, 137, 271, 500, 888]
W_SWEEP       = [0.3, 0.5, 0.8]
TAU_GT_SWEEP  = [20, 75, 150, 250]
TAU_GRID_MS   = [20, 30, 50, 75, 100, 150, 250]

# ── Fixed parameters ───────────────────────────────────────────────────────────
RESULTS_DIR       = Path("models_output/toneclouds_gaussianprf_20260612_0155")
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"
SANITY_FREQ       = "fc572hz"
CF_IDX            = 10
ALPHA             = 4.0
RHO               = 1.0
TR_S              = 1.6
TOTAL_RUN_DUR_S   = 720.0
SIGNAL_DT_S       = 1e-3
RECTIFY           = True
NOISE_SEED        = 42
NOISE_VOXEL       = "mid"
N_RUNS            = 24
SEQ_DUR_S         = 20.0

n_tr_per_run   = int(TOTAL_RUN_DUR_S / TR_S)
n_samp_per_run = int(TOTAL_RUN_DUR_S / SIGNAL_DT_S)


# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID (0 to N_SEEDS*N_W - 1)")
parser.add_argument("--out_dir", type=str, default="tau_recovery_results",
                    help="Directory to save per-task .npz results")
args = parser.parse_args()

ds_idx  = args.task_id // len(W_SWEEP)
w_idx   = args.task_id % len(W_SWEEP)
ds_seed = DESIGN_SEEDS[ds_idx]
w_val   = W_SWEEP[w_idx]

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_ds{ds_seed}_w{w_val}.npz"

print(f"Task {args.task_id}: design_seed={ds_seed}, w={w_val}")
print(f"Output → {out_path}")

# ── Phase 1: load cochlear PSTHs ──────────────────────────────────────────────
print("Loading cochlear PSTHs ...")
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {RESULTS_DIR}/wav*/")

per_seq: dict = {}
cf_hz_used: float = None
cf_list_arr = None

for npz_path in npz_files:
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, CF_IDX)
    if cf_list_arr is None:
        from auditory_prf.utils.result_saver import ResultSaver
        data = ResultSaver(npz_path.parent).load_npz(npz_path.name)
        cf_list_arr = np.asarray(data["cf_list"])
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

active_ids   = sorted([k for k in per_seq if k != TC_SILENCE_SEQ_ID and SANITY_FREQ in k])
n_conds      = len(active_ids)
print(f"  CF={cf_hz_used:.0f} Hz | {n_conds} active sequences")

# ── HRF kernel ─────────────────────────────────────────────────────────────────
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
        onset_s += SEQ_DUR_S + float(rng.randint(5, 16))
    return design


def _build_glm_projector(run_designs: list) -> np.ndarray:
    """Returns projector (X'X)^{-1}X', shape (N_RUNS+n_conds, n_tr_total)."""
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


def forward_run(tau_ms: float, run_design: list) -> np.ndarray:
    return assemble_run_bold(
        per_seq=per_seq, run_design=run_design,
        total_run_dur_s=TOTAL_RUN_DUR_S, hrf_kernel=hrf_kernel,
        cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
        w=w_val, K=None, apply_adaptrans_flag=True,
        rectify=RECTIFY, rho=RHO, tau_ms=tau_ms, cf_range_hz=None,
    )["bold_combined"]


# ── Compute ───────────────────────────────────────────────────────────────────
run_designs = [_make_run_design(ds_seed + i) for i in range(N_RUNS)]
projector   = _build_glm_projector(run_designs)

print(f"Computing {len(TAU_GRID_MS)} × {N_RUNS} forward runs ...")
bold_runs  = {tau: [forward_run(tau, d) for d in run_designs] for tau in TAU_GRID_MS}
pred_betas = {tau: fit_betas(bold_runs[tau], projector)        for tau in TAU_GRID_MS}

r2_noisy_all = {}   # tau_gt -> (n_tau_grid,)
r2_nl_all    = {}

for tau_gt in TAU_GT_SWEEP:
    gt_runs    = bold_runs[tau_gt]
    noise_base = NOISE_SEED + ds_idx * N_RUNS
    noisy_runs = [
        apply_bold_noise(bg, PmNoise(voxel=NOISE_VOXEL, seed=noise_base + i), TR_S)
        for i, bg in enumerate(gt_runs)
    ]
    betas_noisy = fit_betas(noisy_runs, projector)
    betas_nl    = pred_betas[tau_gt]

    r2_noisy_all[tau_gt] = [_r2_betas(pred_betas[tau], betas_noisy) for tau in TAU_GRID_MS]
    r2_nl_all[tau_gt]    = [_r2_betas(pred_betas[tau], betas_nl)    for tau in TAU_GRID_MS]

    best = TAU_GRID_MS[int(np.argmax(r2_noisy_all[tau_gt]))]
    print(f"  τ_GT={tau_gt}ms | best={best}ms {'✓' if best == tau_gt else '✗'}")

# ── Save ──────────────────────────────────────────────────────────────────────
save_dict = {
    "task_id":    args.task_id,
    "ds_idx":     ds_idx,
    "design_seed": ds_seed,
    "w_val":      w_val,
    "cf_hz_used": cf_hz_used,
    "TAU_GRID_MS": TAU_GRID_MS,
    "TAU_GT_SWEEP": TAU_GT_SWEEP,
}
for tau_gt in TAU_GT_SWEEP:
    save_dict[f"r2_noisy_{tau_gt}"] = np.array(r2_noisy_all[tau_gt])
    save_dict[f"r2_nl_{tau_gt}"]    = np.array(r2_nl_all[tau_gt])

np.savez(out_path, **save_dict)
print(f"Saved → {out_path}")
