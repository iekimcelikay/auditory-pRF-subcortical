"""gt_onoff_singlerun_worker.py
===============================
2D extension of gt35_singlerun_worker.py: can we recover separate ON and OFF
AdapTrans time constants (tau_on, tau_off) from single-run noisy BOLD, using
the same noise-repeat/bias methodology as the 1D sweep?

Same architecture as gt35_singlerun_worker.py (real run design via
generate_run_design/apply_run_noise, noiseless GT + N_NOISE_REPEATS=100 noisy
draws per design, single-run GLM, no cross-run pooling, no nuisance
regressors) but the candidate grid is now TAU_GRID_MS x TAU_GRID_MS instead of
TAU_GRID_MS, so each design seed costs len(TAU_GRID_MS)**2 forward passes
instead of len(TAU_GRID_MS).

Task mapping (8 tasks = 4 CFs x 2 GT pairs):
    task_id -> cf_i  = task_id // len(GT_PAIRS)
    task_id -> gt_i  = task_id %  len(GT_PAIRS)

CF_INDICES = [12, 17, 21, 29] -- the 3 real tone-cloud band centers
(549/885/1265Hz) plus the highest CF (2500Hz, idx 29), which showed the worst
1D recovery (recovery_rate 0.29 vs 0.43 at 125Hz) and is therefore the most
informative extreme to re-test with an ON/OFF split.

GT_PAIRS = [(35, 35), (35, 50)] -- a symmetric GT (directly comparable to the
1D GT=35ms result) and an asymmetric GT where ON and OFF genuinely differ.
Both values must be in TAU_GRID_MS for the grid search to be able to land on
them exactly.

Usage (by SLURM):
    python gt_onoff_singlerun_worker.py --task_id <0..7> --out_dir <dir>
"""

import argparse
import numpy as np
from pathlib import Path

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import build_per_seq_trains
from auditory_prf.prf_pipeline.run_assembly import (
    generate_run_design, assemble_run_bold, apply_run_noise, convolve_hrf,
)
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.full_pipeline_toneclouds_adaptrans import (
    BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE, TC_SILENCE_SEQ_ID,
    TONE_ON_MS, ISI_MS, NULL_FRACTION, TRIAL_DURATION_S, OPENING_BLANK_S,
    CHUNK_MARGIN_MS, _make_tonecloud_seq_id_fn,
)
from prf_models.pm_noise import PmNoise

# ── Sweep parameters ───────────────────────────────────────────────────────────
N_DESIGN_SEEDS = 100
DESIGN_SEEDS   = [1000 + i for i in range(N_DESIGN_SEEDS)]  # distinct from BASE_SEED=42
CF_INDICES     = [12, 17, 21, 29]
GT_PAIRS       = [(35, 35), (35, 50)]
TAU_GRID_MS    = [20, 35, 45, 50, 75]  # same grid, both axes

# ── Fixed parameters (must match gt35_singlerun_worker.py) ────────────────────
W_VAL       = 0.8
RHO         = 1.0
ALPHA       = 4.0
TR_S        = 1.6
TOTAL_RUN_DUR_S = 652.8
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
NOISE_VOXEL     = "mid"
N_NOISE_REPEATS = 100
RESULTS_DIR = Path("models_output/toneclouds_gaussianprf_20260619_0114")

n_tr_per_run = int(round(TOTAL_RUN_DUR_S / TR_S))
n_samp_per_run = int(round(TOTAL_RUN_DUR_S / SIGNAL_DT_S))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True)
parser.add_argument("--out_dir", type=str, default="gt_onoff_singlerun_results")
args = parser.parse_args()

CF_IDX  = CF_INDICES[args.task_id // len(GT_PAIRS)]
GT_PAIR = GT_PAIRS[args.task_id % len(GT_PAIRS)]
TAU_ON_GT, TAU_OFF_GT = GT_PAIR
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_cf{CF_IDX:02d}_gt{TAU_ON_GT}-{TAU_OFF_GT}.npz"

print(f"Task {args.task_id}: CF_IDX={CF_IDX}, GT_PAIR={GT_PAIR} ({N_DESIGN_SEEDS} design seeds)")
print(f"Output -> {out_path}")

# ── Phase 1: load cochlear PSTHs (all 3 bands) -- ONCE for this CF ───────────
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
per_seq, cf_hz_used = build_per_seq_trains(npz_files, CF_IDX, ALPHA, TC_SILENCE_SEQ_ID, CHUNK_MARGIN_MS)

seq_id_fn = _make_tonecloud_seq_id_fn(BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE)
n_gaussians = len(BAND_CENTERS_HZ)
stimuli = [(ton, isi, g_idx) for ton, isi in zip(TONE_ON_MS, ISI_MS) for g_idx in range(n_gaussians)]
n_null = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
base_trials = stimuli + [(0, 0, None)] * n_null
active_ids = sorted({seq_id_fn(t[0], t[1], t[2]) for t in stimuli})

hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)


def _build_single_run_projector(run_design):
    """GLM projector for ONE run: intercept + one regressor per condition present."""
    present_ids = sorted({s for s, _ in run_design if s in active_ids})
    X_int = np.ones((n_tr_per_run, 1))
    X_cond = np.zeros((n_tr_per_run, len(present_ids)))
    for seq_id, onset_s in run_design:
        if seq_id not in present_ids:
            continue
        c = present_ids.index(seq_id)
        boxcar = np.zeros(n_samp_per_run)
        i_on = int(round(onset_s / SIGNAL_DT_S))
        i_off = min(i_on + int(TRIAL_DURATION_S / SIGNAL_DT_S), n_samp_per_run)
        boxcar[i_on:i_off] = 1.0
        col = convolve_hrf(boxcar, hrf_kernel, signal_dt=SIGNAL_DT_S, kernel_dt=SIGNAL_DT_S, output_dt=TR_S)
        X_cond[:, c] += col[:n_tr_per_run]
    X_full = np.column_stack([X_int, X_cond])
    return present_ids, np.linalg.pinv(X_full)


def fit_betas_single_run(bold, projector):
    return (projector @ bold)[1:]   # drop the single intercept


def _r2_betas(predicted, observed):
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def forward_run(tau_on_ms, tau_off_ms, run_design, w):
    return assemble_run_bold(
        per_seq=per_seq, run_design=run_design, total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel, cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
        w=w, K=None, apply_adaptrans_flag=True, rectify=RECTIFY, rho=RHO,
        tau_ms=tau_on_ms, tau_ms_off=tau_off_ms,
    )["bold_combined"]


# ── Loop over all 100 design seeds for this one (CF, GT pair) ────────────────
biases_on  = np.empty(N_DESIGN_SEEDS)
biases_off = np.empty(N_DESIGN_SEEDS)
stds_on    = np.empty(N_DESIGN_SEEDS)
stds_off   = np.empty(N_DESIGN_SEEDS)
recovery_rates = np.empty(N_DESIGN_SEEDS)
est_tau_on_per_design  = np.empty((N_DESIGN_SEEDS, N_NOISE_REPEATS), dtype=int)
est_tau_off_per_design = np.empty((N_DESIGN_SEEDS, N_NOISE_REPEATS), dtype=int)
r2_pooled = {(ton, toff): [] for ton in TAU_GRID_MS for toff in TAU_GRID_MS}

for ds_i, design_seed in enumerate(DESIGN_SEEDS):
    run_design = generate_run_design(base_trials, seq_id_fn, trial_duration_s=TRIAL_DURATION_S,
                                     opening_blank_s=OPENING_BLANK_S, iti_range_s=0, seed=design_seed)
    last_onset_s = run_design[-1][1]
    derived = last_onset_s + TRIAL_DURATION_S + OPENING_BLANK_S
    assert np.isclose(derived, TOTAL_RUN_DUR_S), f"derived {derived} != {TOTAL_RUN_DUR_S}"

    bold_run = {(ton, toff): forward_run(ton, toff, run_design, W_VAL)
                for ton in TAU_GRID_MS for toff in TAU_GRID_MS}
    gt_bold = bold_run[GT_PAIR]

    present_ids, projector = _build_single_run_projector(run_design)
    pred_betas = {pair: fit_betas_single_run(bold, projector) for pair, bold in bold_run.items()}

    est_on  = np.empty(N_NOISE_REPEATS, dtype=int)
    est_off = np.empty(N_NOISE_REPEATS, dtype=int)
    for rep in range(N_NOISE_REPEATS):
        noise_model = PmNoise(voxel=NOISE_VOXEL, seed="random")
        noisy_bold = apply_run_noise(gt_bold, noise_model, ds_i, TR_S)
        betas_noisy = fit_betas_single_run(noisy_bold, projector)
        r2_rep = {pair: _r2_betas(pred, betas_noisy) for pair, pred in pred_betas.items()}
        for pair, r2 in r2_rep.items():
            r2_pooled[pair].append(r2)
        best_pair = max(r2_rep, key=r2_rep.get)
        est_on[rep], est_off[rep] = best_pair

    biases_on[ds_i]  = float(est_on.mean())  - TAU_ON_GT
    biases_off[ds_i] = float(est_off.mean()) - TAU_OFF_GT
    stds_on[ds_i]    = float(est_on.std())
    stds_off[ds_i]   = float(est_off.std())
    recovery_rates[ds_i] = float(np.mean((est_on == TAU_ON_GT) & (est_off == TAU_OFF_GT)))
    est_tau_on_per_design[ds_i]  = est_on
    est_tau_off_per_design[ds_i] = est_off

    print(f"  [{ds_i+1}/{N_DESIGN_SEEDS}] seed={design_seed}: "
          f"bias_on={biases_on[ds_i]:+.2f}ms bias_off={biases_off[ds_i]:+.2f}ms "
          f"recovery_rate={recovery_rates[ds_i]:.2f}")

print(f"\nCF_IDX={CF_IDX} ({cf_hz_used:.0f}Hz) GT={GT_PAIR} summary over {N_DESIGN_SEEDS} designs: "
      f"bias_on_mean={biases_on.mean():+.2f}ms bias_off_mean={biases_off.mean():+.2f}ms "
      f"recovery_mean={recovery_rates.mean():.2f}")

save_dict = {
    "task_id": args.task_id, "cf_idx": CF_IDX, "cf_hz_used": cf_hz_used,
    "tau_on_gt": TAU_ON_GT, "tau_off_gt": TAU_OFF_GT,
    "design_seeds": np.array(DESIGN_SEEDS),
    "bias_on": biases_on, "bias_off": biases_off,
    "std_on": stds_on, "std_off": stds_off,
    "recovery_rate": recovery_rates,
    "est_tau_on_per_design": est_tau_on_per_design,
    "est_tau_off_per_design": est_tau_off_per_design,
    "tau_grid_ms": np.array(TAU_GRID_MS),
}
for (ton, toff), vals in r2_pooled.items():
    save_dict[f"r2_ton{ton}_toff{toff}"] = np.array(vals)

np.savez(out_path, **save_dict)
print(f"Saved -> {out_path}")
