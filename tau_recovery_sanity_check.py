"""tau_recovery_sanity_check.py
================================
Sweep over adaptation weight w, ground-truth τ, and random run design seeds
to assess AdapTrans parameter recovery via GLM beta-pattern R².

Architecture
------------
For each design seed:
  - Generate N_RUNS run designs (different trial orderings)
  - Build GLM design matrix X and projector (X'X)^{-1}X'

  For each w value:
    - Precompute candidate BOLD timecourses for all τ in TAU_GRID_MS
      (N_RUNS forward runs per τ — reused across all tau_gt values)

    For each tau_gt:
      - GT BOLD = candidate BOLD at tau_gt (no extra forward calls)
      - Add noise → observed betas via GLM
      - R²(predicted_betas[τ], observed_betas) for all τ candidates

Results are averaged (mean ± std) over N_DESIGN_SEEDS.
Total forward calls: N_DESIGN_SEEDS × len(W_SWEEP) × len(TAU_GRID_MS) × N_RUNS

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python tau_recovery_sanity_check.py
"""

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

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR       = Path("models_output/toneclouds_gaussianprf_20260612_0155")
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"
SANITY_FREQ       = "fc572hz"

# ── Fixed parameters ───────────────────────────────────────────────────────────
CF_IDX          = 10        # ~448 Hz
ALPHA           = 2.0
RHO             = 1.0
TR_S            = 1.6
TOTAL_RUN_DUR_S = 720.0
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
NOISE_SEED      = 42        # base seed; offset by (design_seed_idx * N_RUNS + run_idx)
NOISE_VOXEL     = "mid"
N_RUNS          = 4
SEQ_DUR_S       = 20.0

# ── Sweep parameters ──────────────────────────────────────────────────────────
N_DESIGN_SEEDS = 5
DESIGN_SEEDS   = [42, 137, 271, 500, 888]   # run-design seeds (not noise seeds)
W_SWEEP        = [0.3, 0.5, 0.8]
TAU_GT_SWEEP   = [20, 75, 150, 250]         # must be a subset of TAU_GRID_MS
TAU_GRID_MS    = [20, 30, 50, 75, 100, 150, 250]

n_tr_per_run   = int(TOTAL_RUN_DUR_S / TR_S)
n_samp_per_run = int(TOTAL_RUN_DUR_S / SIGNAL_DT_S)

total_fwd = N_DESIGN_SEEDS * len(W_SWEEP) * len(TAU_GRID_MS) * N_RUNS
print(f"Sweep plan: {N_DESIGN_SEEDS} seeds × {len(W_SWEEP)} w × "
      f"{len(TAU_GRID_MS)} τ × {N_RUNS} runs = {total_fwd} forward calls")

# ── Phase 1: load cochlear PSTHs → per_seq (done once) ────────────────────────
print(f"\nPhase 1: loading cochlear PSTHs from {RESULTS_DIR} (CF index {CF_IDX}) ...")
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {RESULTS_DIR}/wav*/")

per_seq: dict = {}
cf_hz_used: float = None
cf_list_arr: np.ndarray = None

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
            result["onsets_ms"],
            result["offsets_ms"],
            total_dur_ms,
            dt_ms=1.0,
        )

    per_seq[seq_id] = {"train": train, "cf_hz": cf_hz}
    cf_hz_used = cf_hz

active_ids   = sorted([k for k in per_seq if k != TC_SILENCE_SEQ_ID and SANITY_FREQ in k])
durations_ms = [float(sid.split("_dur")[1].split("ms")[0]) for sid in active_ids]
n_conds      = len(active_ids)
print(f"  Loaded {len(per_seq)} sequences | CF={cf_hz_used:.0f} Hz")
print(f"  {n_conds} active sequences at {SANITY_FREQ}: "
      f"durations = {[f'{d:.0f}' for d in durations_ms]} ms")

# ── HRF kernel (built once) ───────────────────────────────────────────────────
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


def _build_glm_projector(run_designs: list) -> tuple:
    """Build (X_full, projector) for a set of N_RUNS run designs.

    Returns
    -------
    X_full      : (n_tr_total, N_RUNS + n_conds)
    projector   : (N_RUNS + n_conds, n_tr_total)  — (X'X)^{-1} X'
    """
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

    X_full    = np.column_stack([X_int, X_cond])
    projector = np.linalg.pinv(X_full)   # (N_RUNS+n_conds, n_tr_total)
    return X_full, projector


def fit_betas(bold_runs: list, projector: np.ndarray) -> np.ndarray:
    """Multi-run GLM → condition betas (n_conds,). Discards run-intercept betas."""
    return (projector @ np.concatenate(bold_runs))[N_RUNS:]


def _r2_betas(predicted: np.ndarray, observed: np.ndarray) -> float:
    """OLS R² of predicted beta pattern vs observed (free scale + offset)."""
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid  = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def forward_run(tau_ms: float, run_design: list, w: float) -> np.ndarray:
    """Return bold_combined (n_TR,) for one run at given τ and w."""
    return assemble_run_bold(
        per_seq=per_seq,
        run_design=run_design,
        total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel,
        cf_hz=cf_hz_used,
        tr_s=TR_S,
        signal_dt_s=SIGNAL_DT_S,
        w=w,
        K=None,
        apply_adaptrans_flag=True,
        rectify=RECTIFY,
        rho=RHO,
        tau_ms=tau_ms,
    )["bold_combined"]


# ── Main sweep: design_seed × w × tau_gt ──────────────────────────────────────
# Accumulate R² arrays: shape (N_DESIGN_SEEDS, len(TAU_GRID_MS)) per (w, tau_gt)
r2_noisy_seeds = {(w, tgt): [] for w in W_SWEEP for tgt in TAU_GT_SWEEP}
r2_nl_seeds    = {(w, tgt): [] for w in W_SWEEP for tgt in TAU_GT_SWEEP}

for ds_i, design_seed in enumerate(DESIGN_SEEDS):
    print(f"\n{'='*60}")
    print(f"Design seed {design_seed}  [{ds_i+1}/{N_DESIGN_SEEDS}]")
    print(f"{'='*60}")

    run_designs = [_make_run_design(design_seed + i) for i in range(N_RUNS)]
    _, projector = _build_glm_projector(run_designs)

    for w_val in W_SWEEP:
        print(f"  w={w_val} — computing {len(TAU_GRID_MS)} × {N_RUNS} forward runs ...")

        bold_runs = {
            tau: [forward_run(tau, d, w_val) for d in run_designs]
            for tau in TAU_GRID_MS
        }
        pred_betas = {tau: fit_betas(bold_runs[tau], projector) for tau in TAU_GRID_MS}

        for tau_gt in TAU_GT_SWEEP:
            gt_runs = bold_runs[tau_gt]

            # Independent noise per (design_seed, run) combination
            noise_base = NOISE_SEED + ds_i * N_RUNS
            noisy_runs = [
                apply_bold_noise(bg, PmNoise(voxel=NOISE_VOXEL, seed=noise_base + i), TR_S)
                for i, bg in enumerate(gt_runs)
            ]
            betas_noisy = fit_betas(noisy_runs, projector)
            betas_nl    = pred_betas[tau_gt]

            r2_n  = [_r2_betas(pred_betas[tau], betas_noisy) for tau in TAU_GRID_MS]
            r2_nl = [_r2_betas(pred_betas[tau], betas_nl)    for tau in TAU_GRID_MS]

            r2_noisy_seeds[(w_val, tau_gt)].append(r2_n)
            r2_nl_seeds[(w_val, tau_gt)].append(r2_nl)

            best_n = TAU_GRID_MS[int(np.argmax(r2_n))]
            ok     = "✓" if best_n == tau_gt else f"✗→{best_n}ms"
            print(f"    τ_GT={tau_gt:3d}ms | noisy best={ok} | "
                  f"R²_range={max(r2_n)-min(r2_n):.4f}")

# ── Aggregate over design seeds ───────────────────────────────────────────────
results = {}
for w_val in W_SWEEP:
    for tau_gt in TAU_GT_SWEEP:
        r2_n  = np.array(r2_noisy_seeds[(w_val, tau_gt)])   # (N_DESIGN_SEEDS, n_tau)
        r2_nl = np.array(r2_nl_seeds[(w_val, tau_gt)])
        results[(w_val, tau_gt)] = {
            'r2_noisy_mean': r2_n.mean(axis=0),
            'r2_noisy_std':  r2_n.std(axis=0),
            'r2_nl_mean':    r2_nl.mean(axis=0),
            'r2_nl_std':     r2_nl.std(axis=0),
        }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"Summary (mean noisy recovery over {N_DESIGN_SEEDS} design seeds)")
print(f"{'w':>6}  {'τ_GT':>6}  {'best τ':>8}  {'correct':>8}  {'R²_range':>10}")
print(f"{'─'*70}")
for w_val in W_SWEEP:
    for tau_gt in TAU_GT_SWEEP:
        r2_mean = results[(w_val, tau_gt)]['r2_noisy_mean']
        best    = TAU_GRID_MS[int(np.argmax(r2_mean))]
        ok      = "✓" if best == tau_gt else f"✗→{best}ms"
        rng     = max(r2_mean) - min(r2_mean)
        print(f"  {w_val:>4}  {tau_gt:>6}  {best:>8}  {ok:>8}  {rng:>10.4f}")

# ── Plot: rows = tau_gt, cols = w ─────────────────────────────────────────────
n_rows = len(TAU_GT_SWEEP)
n_cols = len(W_SWEEP)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
fig.suptitle(
    f"τ recovery — AdapTrans only | β-space R² | CF≈{cf_hz_used:.0f} Hz | α={ALPHA} | "
    f"{N_RUNS} runs × {N_DESIGN_SEEDS} design seeds | noise='{NOISE_VOXEL}' | {SANITY_FREQ}",
    fontsize=9,
)

for col_i, w_val in enumerate(W_SWEEP):
    for row_i, tau_gt in enumerate(TAU_GT_SWEEP):
        ax  = axes[row_i, col_i]
        res = results[(w_val, tau_gt)]
        tg  = np.array(TAU_GRID_MS)

        # Noiseless (green, dashed)
        ax.plot(tg, res['r2_nl_mean'], "o--", color="forestgreen",
                linewidth=1.5, markersize=5, label="noiseless", alpha=0.8)
        ax.fill_between(tg,
                         res['r2_nl_mean'] - res['r2_nl_std'],
                         res['r2_nl_mean'] + res['r2_nl_std'],
                         color="forestgreen", alpha=0.15)

        # Noisy (blue, solid)
        ax.plot(tg, res['r2_noisy_mean'], "o-", color="steelblue",
                linewidth=2, markersize=7, label="noisy (mean±std)")
        ax.fill_between(tg,
                         res['r2_noisy_mean'] - res['r2_noisy_std'],
                         res['r2_noisy_mean'] + res['r2_noisy_std'],
                         color="steelblue", alpha=0.2)

        ax.axvline(x=tau_gt, color="firebrick", linestyle="--",
                   linewidth=1.5, label=f"GT={tau_gt}ms")

        best_n = TAU_GRID_MS[int(np.argmax(res['r2_noisy_mean']))]
        ok     = "✓" if best_n == tau_gt else f"✗→{best_n}ms"
        rng    = max(res['r2_noisy_mean']) - min(res['r2_noisy_mean'])
        ax.set_title(f"w={w_val}  τ_GT={tau_gt}ms  {ok}  (Δ={rng:.4f})", fontsize=8)
        ax.set_xlabel("τ candidate (ms)", fontsize=8)
        ax.set_ylabel("R² (beta pattern)", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(TAU_GRID_MS)
        ax.set_xticklabels([str(t) for t in TAU_GRID_MS], fontsize=6)
        ax.tick_params(axis='y', labelsize=7)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=6)

plt.tight_layout()
out_path = Path("tau_recovery_w_sweep.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved → {out_path.resolve()}")
