"""duration_recovery.py
========================
Parameter-recovery test for the Gaussian duration-tuning model, in isolation
from AdapTrans (``apply_adaptrans_flag=False`` throughout) — see project
discussion: test whether `pref_dur_ms`/`sigma_dur_ms` are recoverable on
their own, before testing the combined duration+AdapTrans model.

Recovered parameters: ``[cf_index, pref_dur_ms, sigma_dur_ms, alpha]`` — no
``w`` (AdapTrans's adaptation weight has no effect when AdapTrans is off;
see ``forward_model.forward_bold_runs``'s docstring) and no ``tau_on``/
``tau_off`` (meaningless without AdapTrans).

Same overall validation logic as ``grid_recovery.py`` (template-matching via
Pearson r against a noiseless candidate grid, Monte Carlo noise draws, on/off
-grid ground-truth sampling, low/mid/high noise sweep) — reuses that
module's generic timing constants, alpha grid, GT-sampling-mode dispatch,
and plotting/table functions (now parameterized by ``param_names``, see
``grid_recovery.py``'s ``plot_recovery``) rather than duplicating them.

Caching note (different from grid_recovery.py): there, ``build_per_seq_trains``
only depends on (alpha, cf_index) and is reused across every (tau_on, tau_off,
w) combo, because AdapTrans+HRF is a later, independent stage. Here, the
duration-Gaussian filter runs *inside* ``build_per_seq_trains`` itself, so
per_seq must be rebuilt for every (alpha, cf_index, pref_dur_ms, sigma_dur_ms)
combination — there is no caching reuse across the duration-parameter axes.

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.duration_recovery
"""

import time
from pathlib import Path

import numpy as np

from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.run_assembly import generate_run_design
from auditory_prf.prf_pipeline.full_pipeline_toneclouds_adaptrans_claude import (
    _make_tonecloud_seq_id_fn,
    TC_SILENCE_SEQ_ID,
    BAND_CENTERS_HZ,
    TOTAL_SEQ_DUR_S,
    STIMULUS_SAMPLE_RATE,
    TONE_ON_MS,
    ISI_MS,
    NULL_FRACTION,
    TRIAL_DURATION_S,
    OPENING_BLANK_S,
    CLOSING_BLANK_S,
    ITI_RANGE_S,
)
from auditory_prf.prf_pipeline.solver.forward_model import (
    load_raw_sequences,
    sharpen_all_sequences,
    build_per_seq_trains,
    forward_bold_runs,
    draw_noisy_measurement,
    spawn_noise_seeds,
)
from auditory_prf.prf_pipeline.solver.grid_recovery import (
    RESULTS_DIR,
    TR_S,
    TOTAL_RUN_DUR_S,
    SIGNAL_DT_S,
    RECTIFY,
    RHO,
    CHUNK_MARGIN_MS,
    N_RUNS_RECOVERY,
    NOISE_VOXELS,
    GT_SAMPLING_MODES,
    ALPHA_MIN,
    ALPHA_MAX,
    ALPHA_K_STEP,
    ALPHA_GRID,
    N_GT_COMBOS,
    N_NOISE_REPEATS,
    MC_SEED,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)
from datetime import datetime

# AdapTrans is off throughout this test -- w/tau_on_ms/tau_off_ms are true
# no-ops in that path (see forward_bold_runs's docstring), these values are
# never actually used.
_UNUSED_W       = 0.0
_UNUSED_TAU_MS  = 0.0

# ── Paths (separate from grid_recovery.py's, own base dir) ───────────────────
RUN_TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR          = Path("duration_recovery_results")
RUN_DIR             = OUTPUT_DIR / RUN_TIMESTAMP
CANDIDATE_GRID_PATH = RUN_DIR / f"candidate_grid_{RUN_TIMESTAMP}.npz"


def recovery_output_paths(gt_mode: str, noise_voxel: str) -> tuple[Path, Path, Path]:
    tag = f"{gt_mode}_{noise_voxel}_{RUN_TIMESTAMP}"
    return (
        RUN_DIR / f"duration_recovery_results_{tag}.npz",
        RUN_DIR / f"duration_recovery_scatter_{tag}.png",
        RUN_DIR / f"duration_recovery_table_{tag}.csv",
    )


def noise_comparison_plot_path(gt_mode: str) -> Path:
    return RUN_DIR / f"duration_recovery_noise_comparison_{gt_mode}_{RUN_TIMESTAMP}.png"


# ── Candidate grid (duration model, no AdapTrans) ────────────────────────────
# pref_dur_ms matches TONE_ON_MS exactly -- the only durations actually
# presented, per project discussion.
PREF_DUR_GRID_MS = list(TONE_ON_MS)

# sigma_dur_ms: continuous-feeling (dense) log2-spaced grid over the same
# 20-400ms range as tau_on/tau_off's grid, per project discussion.
SIGMA_DUR_MIN    = 20.0
SIGMA_DUR_MAX    = 400.0
SIGMA_DUR_K_STEP = 0.1
_sk_min, _sk_max = np.log2(SIGMA_DUR_MIN), np.log2(SIGMA_DUR_MAX)
SIGMA_DUR_GRID_MS = (2 ** np.arange(_sk_min, _sk_max + SIGMA_DUR_K_STEP / 2, SIGMA_DUR_K_STEP)).tolist()

# ── Duration-scaled sigma grid ───────────────────────────────────────────────
# sigma_dur_ms = ratio * pref_dur_ms, clipped at SIGMA_DUR_CAP_MS, so tuning
# width scales with the preferred duration (a 35 ms voxel cannot have a 250 ms
# sigma). The ratio grid is log2-spaced with a step of ~SIGMA_RATIO_K_STEP that
# spans [SIGMA_RATIO_MIN, SIGMA_RATIO_MAX] exactly.
SIGMA_RATIO_MIN    = 0.2
SIGMA_RATIO_MAX    = 0.5
SIGMA_RATIO_K_STEP = 0.1
SIGMA_DUR_CAP_MS   = 200.0


def build_sigma_ratio_grid(
    ratio_min: float = SIGMA_RATIO_MIN,
    ratio_max: float = SIGMA_RATIO_MAX,
    k_step: float = SIGMA_RATIO_K_STEP,
) -> np.ndarray:
    """Log2-spaced sigma/pref_dur ratios. Returns: (n_ratios,)"""
    log2_span = np.log2(ratio_max) - np.log2(ratio_min)
    n_ratios = int(np.ceil(log2_span / k_step - 1e-9)) + 1
    return 2 ** np.linspace(np.log2(ratio_min), np.log2(ratio_max), n_ratios)


def sigma_grid_for_pref_dur(
    pref_dur_ms: float,
    sigma_cap_ms: float = SIGMA_DUR_CAP_MS,
) -> np.ndarray:
    """Candidate sigma_dur_ms values for one preferred duration.

    Returns: (n_sigma,), ascending, clipped at ``sigma_cap_ms`` with the
    duplicates created by clipping removed (so the cap appears once).
    """
    sigma_dur_ms = np.minimum(build_sigma_ratio_grid() * pref_dur_ms, sigma_cap_ms)
    return np.unique(sigma_dur_ms)


def n_scaled_sigma_candidates() -> int:
    """Number of (pref_dur, sigma_dur) pairs in the duration-scaled grid."""
    return sum(len(sigma_grid_for_pref_dur(p)) for p in PREF_DUR_GRID_MS)


def sample_ground_truths_off_grid_scaled(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """Off-grid ground truths with sigma tied to pref_dur.

    pref_dur uniform in range; ratio log-uniform in [SIGMA_RATIO_MIN,
    SIGMA_RATIO_MAX]; sigma = min(ratio * pref_dur, SIGMA_DUR_CAP_MS).

    Returns: (n_combos, 4) columns ``[cf_index, pref_dur_ms, sigma_dur_ms, alpha]``.
    """
    cf_index  = rng.integers(0, n_cf, size=n_combos).astype(float)
    pref_dur  = rng.uniform(PREF_DUR_GRID_MS[0], PREF_DUR_GRID_MS[-1], size=n_combos)
    ratio     = 2 ** rng.uniform(np.log2(SIGMA_RATIO_MIN), np.log2(SIGMA_RATIO_MAX), size=n_combos)
    sigma_dur = np.minimum(ratio * pref_dur, SIGMA_DUR_CAP_MS)
    alpha     = 2 ** rng.uniform(np.log2(ALPHA_MIN), np.log2(ALPHA_MAX), size=n_combos)
    return np.column_stack([cf_index, pref_dur, sigma_dur, alpha])


def sample_ground_truths_on_candidate_grid(
    rng: np.random.Generator, candidate_params: np.ndarray, n_combos: int,
) -> np.ndarray:
    """On-grid ground truths: random rows of the (duration-scaled) candidate grid.

    Drawing rows (not per-axis values) keeps every ground truth an actual
    candidate, so noise-free recovery should be exact.

    Returns: (n_combos, 4) columns ``[cf_index, pref_dur_ms, sigma_dur_ms, alpha]``.
    """
    row_idx = rng.integers(0, len(candidate_params), size=n_combos)
    return candidate_params[row_idx].copy()


def draw_scaled_ground_truths(
    gt_mode: str, rng: np.random.Generator, n_cf: int, n_combos: int, candidate_params: np.ndarray,
) -> np.ndarray:
    """Dispatch on ``gt_mode`` ('on_grid' | 'off_grid') for the duration-scaled grid."""
    assert gt_mode in ("on_grid", "off_grid"), gt_mode
    if gt_mode == "on_grid":
        return sample_ground_truths_on_candidate_grid(rng, candidate_params, n_combos)
    return sample_ground_truths_off_grid_scaled(rng, n_cf, n_combos)


PARAM_NAMES = ["cf_index", "pref_dur_ms", "sigma_dur_ms", "alpha"]


def build_candidate_grid(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate every (cf_index, pref_dur_ms, sigma_dur_ms, alpha) candidate.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4)
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(PREF_DUR_GRID_MS) * len(SIGMA_DUR_GRID_MS) * len(ALPHA_GRID)
    print(f"Building candidate grid: {n_cf} CF x {len(PREF_DUR_GRID_MS)} pref_dur x "
          f"{len(SIGMA_DUR_GRID_MS)} sigma_dur x {len(ALPHA_GRID)} alpha "
          f"= {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        for cf_index in range(n_cf):
            for pref_dur_ms in PREF_DUR_GRID_MS:
                for sigma_dur_ms in SIGMA_DUR_GRID_MS:
                    # Unlike grid_recovery.py, per_seq depends on the
                    # duration params too -- rebuilt every candidate, see
                    # module docstring.
                    per_seq, cf_hz = build_per_seq_trains(
                        raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
                        TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
                        pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
                    )
                    bold_per_run = forward_bold_runs(
                        per_seq, run_designs, hrf_kernel, cf_hz,
                        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
                        _UNUSED_W, _UNUSED_TAU_MS, _UNUSED_TAU_MS, RECTIFY, RHO,
                        apply_adaptrans_flag=False,
                    )
                    template = np.concatenate(bold_per_run)
                    if template_matrix is None:
                        template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                    template_matrix[idx, :] = template
                    candidate_params[idx] = (cf_index, pref_dur_ms, sigma_dur_ms, alpha)
                    idx += 1
        elapsed = time.time() - t_start
        print(f"  alpha={alpha:.3f} done ({idx:,}/{n_candidates:,} candidates, "
              f"{elapsed:.0f}s elapsed)")

    template_matrix -= template_matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
    template_matrix /= np.where(norms == 0, 1.0, norms)

    return candidate_params, template_matrix


def sample_ground_truths_on_grid(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """On-grid ground truths: every parameter sits exactly on a candidate-grid point."""
    cf_index  = rng.integers(0, n_cf, size=n_combos).astype(float)
    pref_dur  = rng.choice(PREF_DUR_GRID_MS, size=n_combos)
    sigma_dur = rng.choice(SIGMA_DUR_GRID_MS, size=n_combos)
    alpha     = rng.choice(ALPHA_GRID, size=n_combos)
    return np.column_stack([cf_index, pref_dur, sigma_dur, alpha])


def sample_ground_truths_off_grid(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """Off-grid ground truths: pref_dur/sigma_dur/alpha drawn continuously within range."""
    cf_index  = rng.integers(0, n_cf, size=n_combos).astype(float)
    pref_dur  = rng.uniform(PREF_DUR_GRID_MS[0], PREF_DUR_GRID_MS[-1], size=n_combos)
    sigma_dur = 2 ** rng.uniform(np.log2(SIGMA_DUR_MIN), np.log2(SIGMA_DUR_MAX), size=n_combos)
    alpha     = 2 ** rng.uniform(np.log2(ALPHA_MIN), np.log2(ALPHA_MAX), size=n_combos)
    return np.column_stack([cf_index, pref_dur, sigma_dur, alpha])


def sample_ground_truths(rng: np.random.Generator, n_cf: int, n_combos: int, gt_mode: str) -> np.ndarray:
    sample_fn = {
        "on_grid":  sample_ground_truths_on_grid,
        "off_grid": sample_ground_truths_off_grid,
    }[gt_mode]
    return sample_fn(rng, n_cf, n_combos)


def run_recovery(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise_voxel: str,
) -> np.ndarray:
    """Monte Carlo recovery loop for one noise level, given already-sampled ground truths.

    Returns
    -------
    recovered_params : np.ndarray, shape (n_gt_combos, n_noise_repeats, 4)
    """
    n_gt_combos = true_params.shape[0]
    recovered_params = np.empty((n_gt_combos, N_NOISE_REPEATS, 4), dtype=float)

    for gt_i, (cf_index_f, pref_dur_ms, sigma_dur_ms, alpha) in enumerate(true_params):
        cf_index = int(round(cf_index_f))
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        per_seq, cf_hz = build_per_seq_trains(
            raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
            TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
            pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
        )
        gt_bold_per_run = forward_bold_runs(
            per_seq, run_designs, hrf_kernel, cf_hz,
            TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
            _UNUSED_W, _UNUSED_TAU_MS, _UNUSED_TAU_MS, RECTIFY, RHO,
            apply_adaptrans_flag=False,
        )

        n_timepoints = sum(len(r) for r in gt_bold_per_run)
        measurements = np.empty((N_NOISE_REPEATS, n_timepoints), dtype=np.float32)
        for rep in range(N_NOISE_REPEATS):
            run_seeds = spawn_noise_seeds(MC_SEED, gt_i, rep, n_runs=N_RUNS_RECOVERY)
            measurements[rep, :] = draw_noisy_measurement(
                gt_bold_per_run, noise_voxel, TR_S, run_seeds,
            )

        measurements -= measurements.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(measurements, axis=1, keepdims=True)
        measurements /= np.where(norms == 0, 1.0, norms)

        r = template_matrix_norm @ measurements.T
        best_idx = np.argmax(r, axis=0)
        recovered_params[gt_i, :, :] = candidate_params[best_idx]

        bias = recovered_params[gt_i].mean(axis=0) - [cf_index, pref_dur_ms, sigma_dur_ms, alpha]
        print(f"  [{noise_voxel}] GT {gt_i+1}/{n_gt_combos}: cf_index={cf_index} "
              f"pref_dur={pref_dur_ms:.1f} sigma_dur={sigma_dur_ms:.1f} alpha={alpha:.3f} "
              f"| bias={np.round(bias, 3)}")

    return recovered_params


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw cochlear PSTH sequences from {RESULTS_DIR} ...")
    raw_sequences, cf_list_hz = load_raw_sequences(RESULTS_DIR)
    n_cf = len(cf_list_hz)
    print(f"  {len(raw_sequences)} sequences loaded | {n_cf} CFs")

    n_gaussians = len(BAND_CENTERS_HZ)
    stimuli = [(ton, isi, g_idx)
               for ton, isi in zip(TONE_ON_MS, ISI_MS)
               for g_idx in range(n_gaussians)]
    n_null = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
    base_trials = stimuli + [(0, 0, None)] * n_null
    seq_id_fn = _make_tonecloud_seq_id_fn(BAND_CENTERS_HZ, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE)
    print(f"  Base trials: {len(stimuli)} active + {n_null} null = {len(base_trials)} total")

    expected_seq_ids = {seq_id_fn(*trial) for trial in stimuli} | {TC_SILENCE_SEQ_ID}
    missing = expected_seq_ids - set(raw_sequences)
    if missing:
        raise ValueError(
            f"{len(missing)} expected seq_ids not found in {RESULTS_DIR} "
            f"(e.g. {sorted(missing)[:3]}). Check BAND_CENTERS_HZ/TONE_ON_MS/ISI_MS "
            f"match the regenerated stimuli."
        )

    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)

    run_design_seeds = [1000 + int(s) for s in
                        np.random.default_rng().integers(0, 100_000, size=N_RUNS_RECOVERY)]
    print(f"  Run-design seeds: {run_design_seeds}")
    run_designs = [
        generate_run_design(
            base_trials, seq_id_fn,
            trial_duration_s=TRIAL_DURATION_S,
            opening_blank_s=OPENING_BLANK_S,
            iti_range_s=ITI_RANGE_S,
            seed=seed,
        )
        for seed in run_design_seeds
    ]
    for run_idx, design in enumerate(run_designs):
        last_onset_s = design[-1][1]
        derived_run_dur_s = last_onset_s + TRIAL_DURATION_S + CLOSING_BLANK_S
        assert np.isclose(derived_run_dur_s, TOTAL_RUN_DUR_S), (
            f"Run {run_idx}: derived run length {derived_run_dur_s:.3f}s != "
            f"TOTAL_RUN_DUR_S={TOTAL_RUN_DUR_S}s."
        )

    candidate_params, template_matrix_norm = build_candidate_grid(
        raw_sequences, cf_list_hz, run_designs, hrf_kernel,
    )
    np.savez(CANDIDATE_GRID_PATH,
             candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
    print(f"Candidate grid saved -> {CANDIDATE_GRID_PATH.resolve()}")

    for gt_mode in GT_SAMPLING_MODES:
        print(f"\n=== GT sampling mode: '{gt_mode}' ===")
        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths(gt_rng, n_cf, N_GT_COMBOS, gt_mode)

        recovered_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning recovery at gt_mode='{gt_mode}', noise_voxel='{noise_voxel}': "
                  f"{N_GT_COMBOS} ground-truth combos x {N_NOISE_REPEATS:,} noise repeats each ...")
            recovered_params = run_recovery(
                raw_sequences, cf_list_hz, run_designs, hrf_kernel,
                candidate_params, template_matrix_norm, true_params, noise_voxel,
            )
            recovered_by_noise[noise_voxel] = recovered_params

            intermediate_path, scatter_plot_path, recovery_table_path = recovery_output_paths(gt_mode, noise_voxel)
            np.savez(intermediate_path, true_params=true_params, recovered_params=recovered_params,
                     param_names=PARAM_NAMES)
            print(f"Recovery results saved -> {intermediate_path.resolve()}")
            plot_recovery(true_params, recovered_params, scatter_plot_path, param_names=PARAM_NAMES)
            save_recovery_table(true_params, recovered_params, recovery_table_path, param_names=PARAM_NAMES)

        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_plot_path(gt_mode),
                               param_names=PARAM_NAMES)


if __name__ == "__main__":
    main()
