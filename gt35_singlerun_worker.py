"""gt35_singlerun_worker.py
===========================
Single worker for the GT tau=35ms single-run recovery sweep across many
different run-design orderings (trial shuffles) AND 10 representative CFs.

Called by the SLURM array job -- each task handles one CF, looping
internally over all 100 design seeds. Phase 1 (loading cochlear PSTHs +
power-law sharpening) only depends on CF_IDX, not on design_seed, so it is
loaded ONCE per task and reused across all 100 design seeds -- avoiding the
100x redundant reload/resharpen that a (design_seed, CF) task grid would do.

Task mapping (10 tasks = 10 CFs):
    task_id -> CF_INDICES[task_id]

Data = ONE noisy realization of a single run's BOLD (no cross-run trial
pooling), matching the supervision-meeting description: "ground truth plus
noise -- that's your data." Candidates = noiseless single-run model
predictions. 100 independent noisy repeats per (design_seed, CF) ->
bias/recovery-rate. Running this across many design seeds (trial orderings)
x CFs checks whether recoverability depends on the arrangement of the run
and on CF, not just one shuffle at one CF.

CF_INDICES = [0,6,10,12,14,17,20,21,25,29] -- the 3 real tone-cloud band
centers (572/885/1322Hz -> idx 12/17/21) plus their +-1 octave neighbours
(idx 6/20, 10/25, 14/29) plus a low-end anchor (idx 0, 125Hz).

DESIGN_SEEDS here are distinct from BASE_SEED (=42, used elsewhere for the
real 24-run experiment design) specifically so a design-seed sweep here can
never collide with the actual per-run seeds used in the real pipeline.

Usage (by SLURM):
    python gt35_singlerun_worker.py --task_id <0..9> --out_dir <dir>
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
CF_INDICES     = [0, 6, 10, 12, 14, 17, 20, 21, 25, 29]

# ── Fixed parameters (must match gt35_singlerun.py) ────────────────────────────
TAU_GRID_MS = [20, 35, 45, 50, 75]
TAU_GT      = 35
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
parser.add_argument("--out_dir", type=str, default="gt35_singlerun_results")
args = parser.parse_args()

CF_IDX = CF_INDICES[args.task_id]
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_cf{CF_IDX:02d}.npz"

print(f"Task {args.task_id}: CF_IDX={CF_IDX} ({N_DESIGN_SEEDS} design seeds)")
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


def forward_run(tau_ms, run_design, w):
    return assemble_run_bold(
        per_seq=per_seq, run_design=run_design, total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel, cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
        w=w, K=None, apply_adaptrans_flag=True, rectify=RECTIFY, rho=RHO, tau_ms=tau_ms,
    )["bold_combined"]


# ── Loop over all 100 design seeds for this one CF ────────────────────────────
biases = np.empty(N_DESIGN_SEEDS)
stds = np.empty(N_DESIGN_SEEDS)
recovery_rates = np.empty(N_DESIGN_SEEDS)
est_tau_per_design = np.empty((N_DESIGN_SEEDS, N_NOISE_REPEATS), dtype=int)
r2_pooled = {tau: [] for tau in TAU_GRID_MS}

for ds_i, design_seed in enumerate(DESIGN_SEEDS):
    run_design = generate_run_design(base_trials, seq_id_fn, trial_duration_s=TRIAL_DURATION_S,
                                     opening_blank_s=OPENING_BLANK_S, iti_range_s=0, seed=design_seed)
    last_onset_s = run_design[-1][1]
    derived = last_onset_s + TRIAL_DURATION_S + OPENING_BLANK_S
    assert np.isclose(derived, TOTAL_RUN_DUR_S), f"derived {derived} != {TOTAL_RUN_DUR_S}"

    bold_run = {tau: forward_run(tau, run_design, W_VAL) for tau in TAU_GRID_MS}
    gt_bold = bold_run[TAU_GT]

    present_ids, projector = _build_single_run_projector(run_design)
    pred_betas = {tau: fit_betas_single_run(bold_run[tau], projector) for tau in TAU_GRID_MS}

    est_tau = np.empty(N_NOISE_REPEATS, dtype=int)
    for rep in range(N_NOISE_REPEATS):
        noise_model = PmNoise(voxel=NOISE_VOXEL, seed="random")
        noisy_bold = apply_run_noise(gt_bold, noise_model, ds_i, TR_S)
        betas_noisy = fit_betas_single_run(noisy_bold, projector)
        r2_rep = {tau: _r2_betas(pred_betas[tau], betas_noisy) for tau in TAU_GRID_MS}
        for tau in TAU_GRID_MS:
            r2_pooled[tau].append(r2_rep[tau])
        est_tau[rep] = TAU_GRID_MS[int(np.argmax([r2_rep[t] for t in TAU_GRID_MS]))]

    biases[ds_i] = float(est_tau.mean()) - TAU_GT
    stds[ds_i] = float(est_tau.std())
    recovery_rates[ds_i] = float(np.mean(est_tau == TAU_GT))
    est_tau_per_design[ds_i] = est_tau

    print(f"  [{ds_i+1}/{N_DESIGN_SEEDS}] seed={design_seed}: "
          f"bias={biases[ds_i]:+.2f}ms recovery_rate={recovery_rates[ds_i]:.2f}")

print(f"\nCF_IDX={CF_IDX} ({cf_hz_used:.0f}Hz) summary over {N_DESIGN_SEEDS} designs: "
      f"bias_mean={biases.mean():+.2f}ms recovery_mean={recovery_rates.mean():.2f}")

save_dict = {
    "task_id": args.task_id, "cf_idx": CF_IDX, "cf_hz_used": cf_hz_used,
    "design_seeds": np.array(DESIGN_SEEDS),
    "bias": biases, "std": stds, "recovery_rate": recovery_rates,
    "est_tau_per_design": est_tau_per_design,
}
for tau in TAU_GRID_MS:
    save_dict[f"r2_tau{tau}"] = np.array(r2_pooled[tau])

np.savez(out_path, **save_dict)
print(f"Saved -> {out_path}")
