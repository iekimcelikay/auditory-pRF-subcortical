"""gt_rho_fulltimecourse_singlerun_worker.py
=============================================
Tests a CST-style (Kim et al. 2024) recovery strategy: keep a single shared
AdapTrans tau (grid-searched, nonlinear) but let the ON/OFF balance (rho,
`bold_combined = rho*bold_on + bold_off`) be a LINEAR GLM coefficient,
estimated by OLS directly against the full continuous BOLD timecourse rather
than a per-condition beta vector.

Rationale (from gt_onoff_singlerun_worker.py's near-zero (tau_on, tau_off)
recovery, and Kim et al.'s CST vs DN-ST comparison): grid-searching two
coupled nonlinear temporal parameters is poorly identified because both
reshape the ON/OFF trains BEFORE the HRF convolution. rho instead scales two
already-HRF-convolved regressors (bold_on, bold_off) -- a purely linear
operation -- so it should be recoverable by ordinary least squares, same as
Kim et al.'s beta_sus/beta_tran channel weights.

Also switches the comparison from per-condition beta-vector R2 (which
collapses each trial to one scalar) to a full-timecourse GLM fit (3 columns:
intercept, bold_on candidate, bold_off candidate) against every TR of the
noisy run -- testing whether keeping trial-by-trial temporal structure
un-flattens the R2-vs-tau curve seen in the beta-vector version.

Same architecture as gt35_singlerun_worker.py otherwise: real run design via
generate_run_design/apply_run_noise, noiseless GT + N_NOISE_REPEATS=100 noisy
draws per design, single-run (no cross-run pooling), no nuisance regressors.

Task mapping (10 tasks = 10 CFs):
    task_id -> CF_INDICES[task_id]

GT: tau=35ms (matches the validated 1D sweep), rho=1.25 (onset-weighted,
distinctly different from assemble_run_bold's default rho=1.0).

Usage (by SLURM):
    python gt_rho_fulltimecourse_singlerun_worker.py --task_id <0..9> --out_dir <dir>
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

# ── Sweep parameters ───────────────────────────────────────────────────────────
N_DESIGN_SEEDS = 100
DESIGN_SEEDS   = [1000 + i for i in range(N_DESIGN_SEEDS)]  # distinct from BASE_SEED=42
CF_INDICES     = [0, 6, 10, 12, 14, 17, 20, 21, 25, 29]
TAU_GRID_MS    = [20, 35, 45, 50, 75]
TAU_GT  = 35
RHO_GT  = 1.25

# ── Fixed parameters (must match gt35_singlerun_worker.py) ────────────────────
W_VAL       = 0.8
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
parser.add_argument("--out_dir", type=str, default="gt_rho_fulltimecourse_singlerun_results")
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


def forward_on_off(tau_ms, run_design):
    """Return (bold_on, bold_off) full-run timecourses at one shared tau."""
    out = assemble_run_bold(
        per_seq=per_seq, run_design=run_design, total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel, cf_hz=cf_hz_used, tr_s=TR_S, signal_dt_s=SIGNAL_DT_S,
        w=W_VAL, K=None, apply_adaptrans_flag=True, rectify=RECTIFY, rho=1.0,
        tau_ms=tau_ms,
    )
    return out["bold_on"], out["bold_off"]


RIDGE_LAMBDA = 10000.0  # bold_on/bold_off are ~0.99 correlated post-HRF (VIF~63x);
# plain OLS makes the individual on/off coefficients (and their ratio, rho_est)
# explode. Ridge-penalizing only the two slope columns (not the intercept)
# stabilizes rho_est to std~0.04 (vs unusable ~38 unregularized), at the cost
# of a small systematic bias (~-0.08 to -0.09, confirmed stable across design
# seeds) -- see conversation history for the lambda sweep that established this.
_RIDGE_P = np.diag([0.0, 1.0, 1.0])


def _fit_full_timecourse(bold_on, bold_off, observed):
    """3-column ridge fit: intercept + bold_on + bold_off, fit against every TR."""
    X = np.column_stack([np.ones(len(observed)), bold_on, bold_off])
    beta = np.linalg.solve(X.T @ X + RIDGE_LAMBDA * _RIDGE_P, X.T @ observed)
    resid = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    a_on, b_off = beta[1], beta[2]
    rho_est = float(a_on / b_off) if b_off != 0 else np.nan
    return r2, rho_est


# ── Loop over all 100 design seeds for this one CF ────────────────────────────
biases_tau  = np.empty(N_DESIGN_SEEDS)
biases_rho  = np.empty(N_DESIGN_SEEDS)
stds_tau    = np.empty(N_DESIGN_SEEDS)
stds_rho    = np.empty(N_DESIGN_SEEDS)
recovery_rates = np.empty(N_DESIGN_SEEDS)
est_tau_per_design = np.empty((N_DESIGN_SEEDS, N_NOISE_REPEATS), dtype=int)
est_rho_per_design = np.empty((N_DESIGN_SEEDS, N_NOISE_REPEATS), dtype=float)
r2_pooled = {tau: [] for tau in TAU_GRID_MS}

for ds_i, design_seed in enumerate(DESIGN_SEEDS):
    run_design = generate_run_design(base_trials, seq_id_fn, trial_duration_s=TRIAL_DURATION_S,
                                     opening_blank_s=OPENING_BLANK_S, iti_range_s=0, seed=design_seed)
    last_onset_s = run_design[-1][1]
    derived = last_onset_s + TRIAL_DURATION_S + OPENING_BLANK_S
    assert np.isclose(derived, TOTAL_RUN_DUR_S), f"derived {derived} != {TOTAL_RUN_DUR_S}"

    onoff_run = {tau: forward_on_off(tau, run_design) for tau in TAU_GRID_MS}
    gt_on, gt_off = onoff_run[TAU_GT]
    gt_bold = RHO_GT * gt_on + gt_off

    est_tau = np.empty(N_NOISE_REPEATS, dtype=int)
    est_rho = np.empty(N_NOISE_REPEATS, dtype=float)
    for rep in range(N_NOISE_REPEATS):
        noise_model = PmNoise(voxel=NOISE_VOXEL, seed="random")
        noisy_bold = apply_run_noise(gt_bold, noise_model, ds_i, TR_S)

        r2_rep = {}
        rho_rep = {}
        for tau in TAU_GRID_MS:
            bold_on, bold_off = onoff_run[tau]
            r2, rho_est = _fit_full_timecourse(bold_on, bold_off, noisy_bold)
            r2_rep[tau] = r2
            rho_rep[tau] = rho_est
            r2_pooled[tau].append(r2)

        best_tau = max(r2_rep, key=r2_rep.get)
        est_tau[rep] = best_tau
        est_rho[rep] = rho_rep[best_tau]

    biases_tau[ds_i] = float(est_tau.mean()) - TAU_GT
    biases_rho[ds_i] = float(np.nanmean(est_rho)) - RHO_GT
    stds_tau[ds_i]   = float(est_tau.std())
    stds_rho[ds_i]   = float(np.nanstd(est_rho))
    recovery_rates[ds_i] = float(np.mean(est_tau == TAU_GT))
    est_tau_per_design[ds_i] = est_tau
    est_rho_per_design[ds_i] = est_rho

    print(f"  [{ds_i+1}/{N_DESIGN_SEEDS}] seed={design_seed}: "
          f"bias_tau={biases_tau[ds_i]:+.2f}ms bias_rho={biases_rho[ds_i]:+.3f} "
          f"recovery_rate={recovery_rates[ds_i]:.2f}")

print(f"\nCF_IDX={CF_IDX} ({cf_hz_used:.0f}Hz) summary over {N_DESIGN_SEEDS} designs: "
      f"bias_tau_mean={biases_tau.mean():+.2f}ms bias_rho_mean={biases_rho.mean():+.3f} "
      f"recovery_mean={recovery_rates.mean():.2f}")

save_dict = {
    "task_id": args.task_id, "cf_idx": CF_IDX, "cf_hz_used": cf_hz_used,
    "tau_gt": TAU_GT, "rho_gt": RHO_GT,
    "design_seeds": np.array(DESIGN_SEEDS),
    "bias_tau": biases_tau, "bias_rho": biases_rho,
    "std_tau": stds_tau, "std_rho": stds_rho,
    "recovery_rate": recovery_rates,
    "est_tau_per_design": est_tau_per_design,
    "est_rho_per_design": est_rho_per_design,
}
for tau in TAU_GRID_MS:
    save_dict[f"r2_tau{tau}"] = np.array(r2_pooled[tau])

np.savez(out_path, **save_dict)
print(f"Saved -> {out_path}")
