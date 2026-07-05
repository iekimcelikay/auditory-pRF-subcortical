"""gt35_fulltimecourse_singlerun_worker.py
===========================================
Re-tests the original gt35_singlerun_worker.py sweep (single shared AdapTrans
tau, GT=35ms, single-run, no cross-run pooling, no nuisance regressors) but
compares candidates against the FULL noisy BOLD timecourse (every TR) instead
of a per-condition beta vector.

This followed two failed intermediate attempts (see conversation history):
1. gt_onoff_singlerun_worker.py: grid-searching (tau_on, tau_off) jointly --
   near-zero recovery, R2 surface almost flat (both nonlinearly reshape the
   ON/OFF trains before the HRF).
2. gt_rho_fulltimecourse_singlerun_worker.py: full-timecourse fit with a
   3-column (intercept, bold_on, bold_off) ridge regression to linearly
   recover rho (ON/OFF balance) alongside tau -- rho itself stabilized with
   ridge (bias ~-0.06, matching a lambda sweep done separately), but tau
   recovery got WORSE (recovery ~0.12 vs ~0.43 in the original beta-based
   sweep) because the 3-column regression's instability leaked into tau
   selection.

This script isolates the full-timecourse comparison from the ON/OFF split:
single tau (rho fixed at 1.0, no ON/OFF decomposition), and the R2 metric is
the same free-scale-offset fit used in the original beta-based sweep, just
applied to the full ~408-TR timecourse instead of an 8-24-point beta vector.
A local check (10 design seeds, CF_IDX=0) showed this recovers BETTER than
the beta-vector version (recovery 0.56 vs 0.43, bias +5.0ms vs +7.5ms) --
apparently the condition-beta averaging's noise reduction was outweighed by
the information it discarded (trial ordering/timing), once the comparison
metric itself stays a simple 2-parameter fit.

Task mapping (10 tasks = 10 CFs):
    task_id -> CF_INDICES[task_id]

Usage (by SLURM):
    python gt35_fulltimecourse_singlerun_worker.py --task_id <0..9> --out_dir <dir>
"""

import argparse
import numpy as np
from pathlib import Path

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.run_assembly import (
    generate_run_design, assemble_run_bold, apply_run_noise,
)
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.full_pipeline_toneclouds_adaptrans import (
    BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE, TC_SILENCE_SEQ_ID,
    TONE_ON_MS, ISI_MS, NULL_FRACTION, TRIAL_DURATION_S, OPENING_BLANK_S,
    _make_tonecloud_seq_id_fn,
)
from prf_models.pm_noise import PmNoise

# ── Sweep parameters (must match gt35_singlerun_worker.py) ────────────────────
N_DESIGN_SEEDS = 100
DESIGN_SEEDS   = [1000 + i for i in range(N_DESIGN_SEEDS)]  # distinct from BASE_SEED=42
CF_INDICES     = [0, 6, 10, 12, 14, 17, 20, 21, 25, 29]
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

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True)
parser.add_argument("--out_dir", type=str, default="gt35_fulltimecourse_singlerun_results")
args = parser.parse_args()

CF_IDX = CF_INDICES[args.task_id]
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_cf{CF_IDX:02d}.npz"

print(f"Task {args.task_id}: CF_IDX={CF_IDX} ({N_DESIGN_SEEDS} design seeds)")
print(f"Output -> {out_path}")

# ── Phase 1: load cochlear PSTHs (all 3 bands) -- ONCE for this CF ───────────
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
per_seq = {}
cf_hz_used = None
for npz_path in npz_files:
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, CF_IDX)
    dt_s = time_axis[1] - time_axis[0]
    total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
    sharpened = apply_powerlaw_population(population_psth, ALPHA)[cf_index, :]
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

seq_id_fn = _make_tonecloud_seq_id_fn(BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE)
n_gaussians = len(BAND_CENTERS_HZ)
stimuli = [(ton, isi, g_idx) for ton, isi in zip(TONE_ON_MS, ISI_MS) for g_idx in range(n_gaussians)]
n_null = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
base_trials = stimuli + [(0, 0, None)] * n_null

hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)


def forward_run(tau_ms, run_design):
    return assemble_run_bold(
        per_seq=per_seq, run_design=run_design, total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel, cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
        w=W_VAL, K=None, apply_adaptrans_flag=True, rectify=RECTIFY, rho=RHO, tau_ms=tau_ms,
    )["bold_combined"]


def _r2_scale_offset(predicted, observed):
    """Free-scale-offset fit against the full timecourse (same metric as the
    original beta-based sweep, just applied to every TR instead of a
    per-condition beta vector)."""
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


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

    bold_run = {tau: forward_run(tau, run_design) for tau in TAU_GRID_MS}
    gt_bold = bold_run[TAU_GT]

    est_tau = np.empty(N_NOISE_REPEATS, dtype=int)
    for rep in range(N_NOISE_REPEATS):
        noise_model = PmNoise(voxel=NOISE_VOXEL, seed="random")
        noisy_bold = apply_run_noise(gt_bold, noise_model, ds_i, TR_S)
        r2_rep = {tau: _r2_scale_offset(bold_run[tau], noisy_bold) for tau in TAU_GRID_MS}
        for tau in TAU_GRID_MS:
            r2_pooled[tau].append(r2_rep[tau])
        est_tau[rep] = max(r2_rep, key=r2_rep.get)

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
