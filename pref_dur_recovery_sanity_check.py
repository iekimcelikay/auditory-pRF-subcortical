"""pref_dur_recovery_sanity_check.py
================================
Grid-search R² sanity check for pref_dur (Gaussian duration filter) parameter recovery.
Compares three CFs nearest to the tone-cloud band centres (572 / 885 / 1322 Hz).

Phase 1 loads cochlear PSTHs for all compared CFs in one pass.
Phase 2 sweeps pref_dur candidates per CF and runs OLS R² against noisy GT.

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python pref_dur_recovery_sanity_check.py
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
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar, gaussian_duration
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, convolve_hrf, SUBCORTICAL_PARAMS
from prf_models.pm_noise import PmNoise, apply_bold_noise

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("models_output/toneclouds_gaussianprf_20260619_0114")
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"

# ── CFs to compare: nearest to each tone-cloud band centre ────────────────────
CF_INDICES = [12, 17, 21]   # 549 Hz / 884 Hz / 1265 Hz

# ── Fixed parameters (match pipeline defaults) ─────────────────────────────────
ALPHA           = 2.0
W               = 0.8
RHO             = 1.0
TR_S            = 1.6
TOTAL_RUN_DUR_S = 720.0
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
NOISE_SEED      = 42
NOISE_VOXEL     = "mid"
TAU_SWEEP       = [50, 100, 200]  # AdapTrans time constants (ms) to compare

# Ground truth pref_dur, fixed sigma, and candidate grid (all in ms)
PREF_DUR_GT_MS   = 75.0
SIGMA_DUR_MS     = 50.0
PREF_DUR_GRID_MS = [35, 50, 75, 100, 150, 250, 488]

# ── Phase 1: load cochlear PSTHs for all CF indices in one pass ───────────────
print(f"Phase 1: loading cochlear PSTHs from {RESULTS_DIR} (CF indices {CF_INDICES}) ...")
npz_files = sorted(RESULTS_DIR.glob("wav*/**/*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found under {RESULTS_DIR}/wav*/")

per_seq_raw_by_cf   = {cf_idx: {} for cf_idx in CF_INDICES}
silence_entry_by_cf = {cf_idx: {} for cf_idx in CF_INDICES}
cf_hz_by_cf         = {cf_idx: None for cf_idx in CF_INDICES}
cf_list_arr         = None

for npz_path in npz_files:
    if cf_list_arr is None:
        from auditory_prf.utils.result_saver import ResultSaver
        data = ResultSaver(npz_path.parent).load_npz(npz_path.name)
        cf_list_arr = np.asarray(data["cf_list"])

    for cf_idx in CF_INDICES:
        population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, cf_idx)
        dt_s         = time_axis[1] - time_axis[0]
        total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
        sharpened    = apply_powerlaw_population(population_psth, ALPHA)[cf_index, :]

        if seq_id == TC_SILENCE_SEQ_ID:
            spont_rate = float(np.mean(sharpened))
            silence_entry_by_cf[cf_idx] = {
                "train": np.full(int(round(total_dur_ms)), spont_rate),
                "cf_hz": cf_hz,
            }
        else:
            result, tone_dur_ms, _ = chunk_from_id(sharpened, time_axis, seq_id)
            per_seq_raw_by_cf[cf_idx][seq_id] = {
                "mean_rates_on": [np.mean(c) for c in result["chunks"]],
                "onsets_ms":     result["onsets_ms"],
                "offsets_ms":    result["offsets_ms"],
                "total_dur_ms":  total_dur_ms,
                "tone_dur_ms":   tone_dur_ms,
                "cf_hz":         cf_hz,
            }
        cf_hz_by_cf[cf_idx] = cf_hz

for cf_idx in CF_INDICES:
    print(f"  CF {cf_idx:02d} ({cf_hz_by_cf[cf_idx]:.0f} Hz): "
          f"{len(per_seq_raw_by_cf[cf_idx])} active sequences")

# Print Gaussian weights at GT pref_dur for the first CF (same durations for all)
print(f"\n  Gaussian weight per duration (pref_dur={PREF_DUR_GT_MS:.0f} ms, sigma={SIGMA_DUR_MS:.0f} ms):")
seen_durs = {}
for seq_id, raw in sorted(per_seq_raw_by_cf[CF_INDICES[0]].items()):
    d = raw["tone_dur_ms"]
    if d not in seen_durs:
        weight = gaussian_duration(d, PREF_DUR_GT_MS, SIGMA_DUR_MS)
        seen_durs[d] = weight
        print(f"    dur={d:6.1f} ms  weight={weight:.5f}")

# ── Build shared run designs (seq_ids are CF-independent) ─────────────────────
N_RUNS     = 24
SEQ_DUR_S  = 20.0
BASE_SEED  = 42
active_ids = list(per_seq_raw_by_cf[CF_INDICES[0]].keys())


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


run_designs = [_make_run_design(BASE_SEED + i) for i in range(N_RUNS)]
print(f"\nRun designs: {N_RUNS} runs × ~{np.mean([len(d) for d in run_designs]):.0f} trials each")

# ── HRF kernel ─────────────────────────────────────────────────────────────────
hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)


# ── Per-CF helpers ─────────────────────────────────────────────────────────────
def _build_per_seq(cf_idx: int, pref_dur_ms: float) -> dict:
    out = {TC_SILENCE_SEQ_ID: silence_entry_by_cf[cf_idx]}
    for seq_id, raw in per_seq_raw_by_cf[cf_idx].items():
        amplitudes = [
            apply_duration_gaussian_scalar(m, raw["tone_dur_ms"], pref_dur_ms, SIGMA_DUR_MS)
            for m in raw["mean_rates_on"]
        ]
        train = build_prf_boxcar_train(
            amplitudes, raw["onsets_ms"], raw["offsets_ms"], raw["total_dur_ms"],
        )
        out[seq_id] = {"train": train, "cf_hz": raw["cf_hz"]}
    return out


def _forward_run(cf_idx: int, pref_dur_ms: float, run_design: list, tau_ms: float) -> np.ndarray:
    return assemble_run_bold(
        per_seq=_build_per_seq(cf_idx, pref_dur_ms),
        run_design=run_design,
        total_run_dur_s=TOTAL_RUN_DUR_S,
        hrf_kernel=hrf_kernel,
        cf_hz=cf_hz_by_cf[cf_idx],
        tr_s=TR_S,
        signal_dt_s=SIGNAL_DT_S,
        w=W, K=None,
        apply_adaptrans_flag=True,
        rectify=RECTIFY,
        rho=RHO,
        tau_ms=tau_ms,
    )["bold_combined"]


# ── GLM helpers (beta-space recovery) ─────────────────────────────────────────
n_tr_per_run   = int(TOTAL_RUN_DUR_S / TR_S)
n_samp_per_run = int(TOTAL_RUN_DUR_S / SIGNAL_DT_S)


def _build_glm_projector(run_designs: list) -> np.ndarray:
    """(X'X)^{-1}X' for run intercepts + per-condition HRF regressors."""
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
    X = np.column_stack([np.ones(len(observed)), predicted])
    beta, _ = np.linalg.lstsq(X, observed, rcond=None)[:2]
    resid  = observed - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


print(f"\nBuilding GLM projector ({N_RUNS} runs × {n_tr_per_run} TRs, {len(active_ids)} conditions) ...")
projector = _build_glm_projector(run_designs)

# ── Phase 2: beta-space recovery per tau × CF — one figure per tau ────────────
for tau_ms in TAU_SWEEP:
    print(f"\n{'═' * 55}")
    print(f"  τ = {tau_ms} ms")
    print(f"{'═' * 55}")

    results_by_cf = {}

    for cf_idx in CF_INDICES:
        cf_hz = cf_hz_by_cf[cf_idx]
        print(f"\n── CF {cf_idx:02d} ({cf_hz:.0f} Hz) ──────────────────────────────")

        bold_gt_runs    = [_forward_run(cf_idx, PREF_DUR_GT_MS, d, tau_ms) for d in run_designs]
        bold_noisy_runs = [
            apply_bold_noise(bg, PmNoise(voxel=NOISE_VOXEL, seed=NOISE_SEED + i), TR_S)
            for i, bg in enumerate(bold_gt_runs)
        ]
        betas_gt    = _fit_betas(bold_gt_runs,    projector)
        betas_noisy = _fit_betas(bold_noisy_runs, projector)
        snr = np.std(betas_gt) / np.std(betas_noisy - betas_gt)
        print(f"  beta SNR≈{snr:.2f}")

        betas_candidates = {
            pd: _fit_betas([_forward_run(cf_idx, pd, d, tau_ms) for d in run_designs], projector)
            for pd in PREF_DUR_GRID_MS
        }

        r2_noisy, r2_noiseless = [], []
        print(f"  {'pref_dur':>10}  {'R²(noisy)':>10}  {'R²(noiseless)':>14}")
        print(f"  {'─' * 38}")
        for pd in PREF_DUR_GRID_MS:
            rn = _r2_betas(betas_candidates[pd], betas_noisy)
            rc = _r2_betas(betas_candidates[pd], betas_gt)
            r2_noisy.append(rn)
            r2_noiseless.append(rc)
            marker = " ◄ GT" if pd == PREF_DUR_GT_MS else ""
            print(f"  {pd:>10}  {rn:>10.4f}  {rc:>14.5f}{marker}")

        results_by_cf[cf_idx] = {
            "r2_noisy":     r2_noisy,
            "r2_noiseless": r2_noiseless,
            "snr":          snr,
            "cf_hz":        cf_hz,
        }

    # ── Plot: 2 rows × 3 cols for this tau ────────────────────────────────────
    fig, axes = plt.subplots(2, len(CF_INDICES), figsize=(5 * len(CF_INDICES), 9),
                              sharex=True, sharey="row")
    fig.suptitle(
        f"pref_dur recovery (beta-space) — τ={tau_ms} ms  |  "
        f"GT={PREF_DUR_GT_MS:.0f} ms  |  σ={SIGMA_DUR_MS:.0f} ms  |  "
        f"noise='{NOISE_VOXEL}'  |  {N_RUNS} runs",
        fontsize=11,
    )

    for col, cf_idx in enumerate(CF_INDICES):
        res   = results_by_cf[cf_idx]
        cf_hz = res["cf_hz"]
        tg    = np.array(PREF_DUR_GRID_MS)

        for row, (r2_vals, label, color) in enumerate([
            (res["r2_noiseless"], "Noiseless", "forestgreen"),
            (res["r2_noisy"],     f"Noisy (beta SNR≈{res['snr']:.2f})", "steelblue"),
        ]):
            ax = axes[row, col]
            ax.plot(tg, r2_vals, "o-", color=color, linewidth=2, markersize=7)
            ax.axvline(x=PREF_DUR_GT_MS, color="firebrick", linestyle="--",
                       linewidth=1.5, label=f"GT={PREF_DUR_GT_MS:.0f} ms")
            best = PREF_DUR_GRID_MS[int(np.argmax(r2_vals))]
            ok   = "✓" if best == PREF_DUR_GT_MS else f"✗→{best} ms"
            ax.set_title(f"CF {cf_idx:02d}  {cf_hz:.0f} Hz\n{label}  {ok}", fontsize=9)
            ax.set_xscale("log")
            ax.set_xticks(PREF_DUR_GRID_MS)
            ax.set_xticklabels([str(t) for t in PREF_DUR_GRID_MS], fontsize=7)
            ax.set_ylabel("R² (beta space)", fontsize=9)
            if row == 1:
                ax.set_xlabel("pref_dur (ms)", fontsize=9)
            if col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"pref_dur_recovery_sanity_check_tau{tau_ms:.0f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path.resolve()}")
