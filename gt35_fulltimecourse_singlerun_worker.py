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

Noise sweep: --noise_voxels selects which PmNoise preset(s) to test (default
all four: noiseless 'none' plus 'low'/'mid'/'high'). All levels are computed
per design seed within the same job -- the noiseless forward model (bold_run)
doesn't depend on noise level, so it's built once per design seed and reused.
'none' is deterministic (gt_bold vs itself), so it only needs a single pass
instead of the 100 noise repeats used for the other levels. Results are saved
per-level with a level suffix, e.g. "bias_low", "recovery_rate_high".

Task mapping (10 tasks = 10 CFs):
    task_id -> CF_INDICES[task_id]

Usage (by SLURM):
    python gt35_fulltimecourse_singlerun_worker.py --task_id <0..9> --out_dir <dir> \
        --noise_voxels none low mid high
"""

import argparse
import numpy as np
from pathlib import Path

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import build_per_seq_trains
from auditory_prf.prf_pipeline.run_assembly import (
    generate_run_design, assemble_run_bold, apply_run_noise, parse_noise_seed_arg,
)
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.full_pipeline_toneclouds_adaptrans_claude import (
    BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE, TC_SILENCE_SEQ_ID,
    TONE_ON_MS, ISI_MS, NULL_FRACTION, TRIAL_DURATION_S, OPENING_BLANK_S,
    CHUNK_MARGIN_MS, _make_tonecloud_seq_id_fn,
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
N_NOISE_REPEATS = 100
RESULTS_DIR = Path("models_output/toneclouds_gaussianprf_20260619_0114")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True)
parser.add_argument("--out_dir", type=str, default="gt35_fulltimecourse_singlerun_results")
parser.add_argument("--noise_voxels", nargs="+", default=["none", "low", "mid", "high"],
                    choices=["none", "low", "mid", "high"],
                    help="BOLD noise preset(s) to sweep (PmNoise voxel level). "
                         "'none' = noiseless pass (single deterministic repeat).")
parser.add_argument("--noise_seed", type=str, default="random",
                    help="PmNoise seed shared across noise levels: 'random' (default, "
                         "matches the original single-noise-level sweep), an int for "
                         "reproducible noise (each of the 100 repeats is offset by its "
                         "index), or 'none'/'nonoise'.")
args = parser.parse_args()

CF_IDX = CF_INDICES[args.task_id]
NOISE_VOXELS = args.noise_voxels
NOISE_SEED = parse_noise_seed_arg(args.noise_seed)
N_REPS_BY_LEVEL = {level: (1 if level == "none" else N_NOISE_REPEATS) for level in NOISE_VOXELS}
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"task_{args.task_id:02d}_cf{CF_IDX:02d}.npz"

print(f"Task {args.task_id}: CF_IDX={CF_IDX} ({N_DESIGN_SEEDS} design seeds), "
      f"noise_voxels={NOISE_VOXELS}")
print(f"Output -> {out_path}")

# ── Phase 1: load cochlear PSTHs (all 3 bands) -- ONCE for this CF ───────────
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
per_seq, cf_hz_used = build_per_seq_trains(npz_files, CF_IDX, ALPHA, TC_SILENCE_SEQ_ID, CHUNK_MARGIN_MS)

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
    # // NOTE: I'm not sure about this measure
    
    """Free-scale-offset fit against the full timecourse (same metric as the
    original beta-based sweep, just applied to every TR instead of a
    per-condition beta vector)."""
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ── Loop over all 100 design seeds for this one CF, x each noise level ───────
biases = {level: np.empty(N_DESIGN_SEEDS) for level in NOISE_VOXELS}
stds = {level: np.empty(N_DESIGN_SEEDS) for level in NOISE_VOXELS}
recovery_rates = {level: np.empty(N_DESIGN_SEEDS) for level in NOISE_VOXELS}
est_tau_per_design = {
    level: np.empty((N_DESIGN_SEEDS, N_REPS_BY_LEVEL[level]), dtype=int)
    for level in NOISE_VOXELS
}
r2_pooled = {level: {tau: [] for tau in TAU_GRID_MS} for level in NOISE_VOXELS}

for ds_i, design_seed in enumerate(DESIGN_SEEDS):
    run_design = generate_run_design(base_trials, seq_id_fn, trial_duration_s=TRIAL_DURATION_S,
                                     opening_blank_s=OPENING_BLANK_S, iti_range_s=0, seed=design_seed)
    last_onset_s = run_design[-1][1]
    derived = last_onset_s + TRIAL_DURATION_S + OPENING_BLANK_S
    assert np.isclose(derived, TOTAL_RUN_DUR_S), f"derived {derived} != {TOTAL_RUN_DUR_S}"

    # Noiseless forward model doesn't depend on noise level -- build once, reuse below.
    bold_run = {tau: forward_run(tau, run_design) for tau in TAU_GRID_MS}
    gt_bold = bold_run[TAU_GT]

    summary_bits = []
    for level in NOISE_VOXELS:
        n_reps = N_REPS_BY_LEVEL[level]
        est_tau = np.empty(n_reps, dtype=int)
        for rep in range(n_reps):
            if level == "none":
                noisy_bold = gt_bold
            else:
                noise_model = PmNoise(voxel=level, seed=NOISE_SEED)
                noisy_bold = apply_run_noise(gt_bold, noise_model, rep, TR_S)
            r2_rep = {tau: _r2_scale_offset(bold_run[tau], noisy_bold) for tau in TAU_GRID_MS}
            for tau in TAU_GRID_MS:
                r2_pooled[level][tau].append(r2_rep[tau])
            est_tau[rep] = max(r2_rep, key=r2_rep.get)

        biases[level][ds_i] = float(est_tau.mean()) - TAU_GT
        stds[level][ds_i] = float(est_tau.std())
        recovery_rates[level][ds_i] = float(np.mean(est_tau == TAU_GT))
        est_tau_per_design[level][ds_i] = est_tau
        summary_bits.append(f"{level}: bias={biases[level][ds_i]:+.2f}ms "
                             f"recovery={recovery_rates[level][ds_i]:.2f}")

    print(f"  [{ds_i+1}/{N_DESIGN_SEEDS}] seed={design_seed}: " + " | ".join(summary_bits))

for level in NOISE_VOXELS:
    print(f"\nCF_IDX={CF_IDX} ({cf_hz_used:.0f}Hz) noise={level} summary over {N_DESIGN_SEEDS} designs: "
          f"bias_mean={biases[level].mean():+.2f}ms recovery_mean={recovery_rates[level].mean():.2f}")

save_dict = {
    "task_id": args.task_id, "cf_idx": CF_IDX, "cf_hz_used": cf_hz_used,
    "design_seeds": np.array(DESIGN_SEEDS),
    "noise_voxels": np.array(NOISE_VOXELS),
}
for level in NOISE_VOXELS:
    save_dict[f"bias_{level}"] = biases[level]
    save_dict[f"std_{level}"] = stds[level]
    save_dict[f"recovery_rate_{level}"] = recovery_rates[level]
    save_dict[f"est_tau_per_design_{level}"] = est_tau_per_design[level]
    for tau in TAU_GRID_MS:
        save_dict[f"r2_tau{tau}_{level}"] = np.array(r2_pooled[level][tau])

np.savez(out_path, **save_dict)
print(f"Saved -> {out_path}")
