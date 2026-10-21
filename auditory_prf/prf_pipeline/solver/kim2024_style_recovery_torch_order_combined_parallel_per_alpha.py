"""kim2024_style_recovery_torch_order_combined_parallel_per_alpha.py
======================================================================
Same combined AdapTrans + duration-selectivity recovery procedure as
``kim2024_style_recovery_torch_order_combined.py``, this variant
parallelizes the candidate-grid build with a fresh ``multiprocessing.Pool``
per alpha, same rationale as ``grid_recovery_parallel_per_alpha.py``.

Per-worker caching: ``per_seq`` depends on (cf_index, alpha, pref_dur_ms,
sigma_dur_ms) -- NOT on tau_ms/w, which only affect the later AdapTrans+HRF
stage -- so each worker caches ``per_seq`` keyed by (cf_index, pref_dur_ms,
sigma_dur_ms) within one alpha's pool, reused across every (tau, w)
combination. Tau is an absolute grid value (no CF anchor), carried directly
in the task tuple.

Usage (from project root, request multiple CPUs via SLURM — see
slurm_scripts/batch_kim2024_style_recovery_torch_order_combined_per_alpha.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined_parallel_per_alpha
"""

import os
import time
from multiprocessing import Pool

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
from auditory_prf.prf_pipeline.solver.forward_model import load_raw_sequences, forward_bold_runs
from auditory_prf.prf_pipeline.solver.forward_model_torch_order_fixed_reference import (
    build_per_seq_trains_torch_order_fixed_reference,
)
from auditory_prf.prf_pipeline.solver.reference_calibration import build_population_pool
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
    N_DESIGN_SEEDS,
    DESIGN_SEEDS,
    MC_SEED,
    N_NOISE_REPEATS,
    ALPHA_GRID,
)
from auditory_prf.prf_pipeline.solver.duration_recovery import PREF_DUR_GRID_MS
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    N_VOXELS,
    compute_accuracy_correlations,
    save_accuracy_table,
    plot_accuracy_comparison,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined import (
    PARAM_NAMES,
    K_SIGMA_GRID,
    TAU_GRID_MS,
    W_GRID,
    RUN_DIR,
    candidate_grid_path,
    recovery_output_paths,
    noise_comparison_plot_path,
    accuracy_table_path,
    accuracy_comparison_plot_path,
    sample_ground_truths_off_grid,
)
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# Ground-truth sampling: "on_grid" = random candidate-grid rows (noise-free recovery should be exact),
# "off_grid" = continuous draws (also exposes grid-quantization error).
GT_MODE = "off_grid"


def draw_ground_truths(gt_mode: str, rng: np.random.Generator, n_cf: int, cf_list_hz: np.ndarray,
                       cf_range_hz: tuple, n_voxels: int, candidate_params: np.ndarray) -> np.ndarray:
    """Draw ground-truth (cf_index, pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w) rows.

    Returns
    -------
    np.ndarray, shape (n_voxels, 6)
    """
    assert gt_mode in ("on_grid", "off_grid"), gt_mode
    if gt_mode == "on_grid":
        rows = rng.integers(0, len(candidate_params), size=n_voxels)
        return candidate_params[rows].copy()
    return sample_ground_truths_off_grid(rng, n_cf, cf_list_hz, cf_range_hz, n_voxels)


# SLURM array: task i runs design seed DESIGN_SEEDS[i] only (see the .slurm's --array).
if "SLURM_ARRAY_TASK_ID" in os.environ:
    DESIGN_SEEDS = [DESIGN_SEEDS[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    N_DESIGN_SEEDS = len(DESIGN_SEEDS)

_worker: dict = {}


def _init_worker(raw_sequences: dict, cf_list_hz: np.ndarray, alpha: float,
                  population_pool: np.ndarray, cf_range_hz: tuple,
                  run_designs: list, hrf_kernel: np.ndarray) -> None:
    _worker["raw_sequences"] = raw_sequences
    _worker["cf_list_hz"] = cf_list_hz
    _worker["alpha"] = alpha
    _worker["population_pool"] = population_pool
    _worker["cf_range_hz"] = cf_range_hz
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel
    _worker["per_seq_cache"] = {}


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, pref_dur_ms, sigma_dur_ms, tau_ms, w) candidate."""
    cf_index, pref_dur_ms, sigma_dur_ms, tau_ms, w = params
    cf_index = int(cf_index)

    # Single-entry cache: tasks arrive in chunks of exactly one
    # (cf_index, pref_dur_ms, sigma_dur_ms) key (chunksize = n_tau * n_w, see
    # build_candidate_grid_per_alpha), so a key is never revisited after its
    # chunk ends. The previous unbounded dict never got a cross-chunk hit and
    # grew until the pool closed (job 7687058 hit 33.5 GB MaxRSS -> OOM).
    cache_key = (cf_index, pref_dur_ms, sigma_dur_ms)
    if _worker["per_seq_cache"].get("key") != cache_key:
        _worker["per_seq_cache"] = {
            "key": cache_key,
            "value": build_per_seq_trains_torch_order_fixed_reference(
                _worker["raw_sequences"], cf_index, _worker["cf_list_hz"], _worker["alpha"],
                TC_SILENCE_SEQ_ID, _worker["population_pool"], chunk_margin_ms=CHUNK_MARGIN_MS,
                pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
            ),
        }
    per_seq, cf_hz = _worker["per_seq_cache"]["value"]

    bold_per_run = forward_bold_runs(
        per_seq, _worker["run_designs"], _worker["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, w, tau_ms, tau_ms, RECTIFY, RHO,
    )
    return np.concatenate(bold_per_run).astype(np.float32)


def build_candidate_grid_per_alpha(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    population_pool: np.ndarray,
    cf_range_hz: tuple,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of
    ``kim2024_style_recovery_torch_order_combined.build_candidate_grid``."""
    n_cf = len(cf_list_hz)
    n_candidates = (n_cf * len(PREF_DUR_GRID_MS) * len(K_SIGMA_GRID)
                    * len(ALPHA_GRID) * len(TAU_GRID_MS) * len(W_GRID))
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha, "
          f"torch order, combined AdapTrans+duration): {n_cf} CF x {len(PREF_DUR_GRID_MS)} pref_dur x "
          f"{len(K_SIGMA_GRID)} k_sigma x {len(ALPHA_GRID)} alpha x {len(TAU_GRID_MS)} tau x "
          f"{len(W_GRID)} w = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 6), dtype=float)
    template_matrix = None
    chunk_size = len(TAU_GRID_MS) * len(W_GRID)   # one (cf_index, pref_dur, k_sigma)'s worth
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        candidate_param_list = [
            (cf_index, pref_dur_ms, k_sigma * pref_dur_ms, tau_ms, w)
            for cf_index in range(n_cf)
            for pref_dur_ms in PREF_DUR_GRID_MS
            for k_sigma in K_SIGMA_GRID
            for tau_ms in TAU_GRID_MS
            for w in W_GRID
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, cf_list_hz, alpha, population_pool, cf_range_hz,
                            run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, pref_dur_ms, sigma_dur_ms, tau_ms, w = candidate_param_list[local_i]
                template_matrix[idx, :] = template
                candidate_params[idx] = (cf_index, pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w)
                idx += 1

        elapsed = time.time() - t_start
        print(f"  alpha={alpha:.3f} done ({idx:,}/{n_candidates:,} candidates, "
              f"{elapsed:.0f}s elapsed)")

    template_matrix -= template_matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
    template_matrix /= np.where(norms == 0, 1.0, norms)

    return candidate_params, template_matrix


def _recover_one_voxel(v_i: int, true_row: np.ndarray, state: dict) -> np.ndarray:
    """Worker task: all MC noise repeats for one ground-truth voxel.

    Returns
    -------
    np.ndarray, shape (N_NOISE_REPEATS, 6)
        Recovered (cf_index, pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w) per repeat.
    """
    cf_index_f, pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w = true_row
    cf_index = int(round(cf_index_f))
    per_seq, cf_hz = build_per_seq_trains_torch_order_fixed_reference(
        state["raw_sequences"], cf_index, state["cf_list_hz"], alpha, TC_SILENCE_SEQ_ID,
        state["population_pool"], chunk_margin_ms=CHUNK_MARGIN_MS,
        pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
    )
    gt_bold_per_run = forward_bold_runs(
        per_seq, state["run_designs"], state["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, w, tau_ms, tau_ms, RECTIFY, RHO,
    )

    n_timepoints = sum(len(r) for r in gt_bold_per_run)
    measurements = np.empty((N_NOISE_REPEATS, n_timepoints), dtype=np.float32)
    for rep in range(N_NOISE_REPEATS):
        run_seeds = spawn_noise_seeds(MC_SEED, v_i, rep, n_runs=N_RUNS_RECOVERY)
        measurements[rep, :] = draw_noisy_measurement(
            gt_bold_per_run, state["noise_voxel"], TR_S, run_seeds,
        )

    best_idx = best_match_indices(state["template_matrix_norm"], measurements)
    return state["candidate_params"][best_idx]


def run_recovery(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    population_pool: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise_voxel: str,
    n_workers: int = N_WORKERS,
) -> np.ndarray:
    """Voxel-parallel version of
    ``kim2024_style_recovery_torch_order_combined.run_recovery`` (same seeds,
    same per-voxel procedure; repeats are scored in memory-bounded chunks --
    the unchunked (483,840 x 10,000) float32 score matrix alone is ~19 GB).

    Returns
    -------
    np.ndarray, shape (n_voxels, N_NOISE_REPEATS, 6)
    """
    n_voxels = true_params.shape[0]
    recovered_params = np.empty((n_voxels, N_NOISE_REPEATS, 6), dtype=float)
    state = dict(
        raw_sequences=raw_sequences, cf_list_hz=cf_list_hz, run_designs=run_designs,
        hrf_kernel=hrf_kernel, population_pool=population_pool,
        candidate_params=candidate_params, template_matrix_norm=template_matrix_norm,
        noise_voxel=noise_voxel,
    )
    for v_i, voxel_recovered in enumerate(map_voxels(_recover_one_voxel, true_params, state, n_workers)):
        recovered_params[v_i] = voxel_recovered
        cf_index_f, pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w = true_params[v_i]
        bias = voxel_recovered.mean(axis=0) - [int(round(cf_index_f)), pref_dur_ms, sigma_dur_ms, alpha, tau_ms, w]
        print(f"  [{noise_voxel}] voxel {v_i+1}/{n_voxels}: true cf_index={int(round(cf_index_f))} "
              f"pref_dur={pref_dur_ms:.1f} sigma_dur={sigma_dur_ms:.1f} alpha={alpha:.3f} "
              f"tau={tau_ms:.1f} w={w:.3f} | bias={np.round(bias, 3)}", flush=True)
    return recovered_params


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw cochlear PSTH sequences from {RESULTS_DIR} ...")
    raw_sequences, cf_list_hz = load_raw_sequences(RESULTS_DIR)
    n_cf = len(cf_list_hz)
    cf_range_hz = (float(cf_list_hz.min()), float(cf_list_hz.max()))
    print(f"  {len(raw_sequences)} sequences loaded | {n_cf} CFs | cf_range_hz={cf_range_hz}")

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

    print("Building population pool (all CFs, all active sequences + silence) ...")
    population_pool = build_population_pool(
        raw_sequences, cf_list_hz, TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
    )
    print(f"  population_pool shape: {population_pool.shape}")

    print(f"\nSweeping {N_DESIGN_SEEDS} design seeds: {DESIGN_SEEDS}")
    for design_seed in DESIGN_SEEDS:
        print(f"\n######## Design seed {design_seed} ########")

        run_design_seeds = [1000 + int(s) for s in
                            np.random.default_rng(design_seed).integers(0, 100_000, size=N_RUNS_RECOVERY)]
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

        candidate_params, template_matrix_norm = build_candidate_grid_per_alpha(
            raw_sequences, cf_list_hz, run_designs, hrf_kernel, population_pool, cf_range_hz,
            n_workers=N_WORKERS,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = draw_ground_truths(GT_MODE, gt_rng, n_cf, cf_list_hz, cf_range_hz, N_VOXELS,
                                         candidate_params)
        assert true_params.shape == (N_VOXELS, 6), true_params.shape

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning recovery at design_seed={design_seed}, noise_voxel='{noise_voxel}': "
                  f"{N_VOXELS} independent voxels x {N_NOISE_REPEATS:,} MC noise repeats each ...")
            recovered_params = run_recovery(
                raw_sequences, cf_list_hz, run_designs, hrf_kernel, population_pool,
                candidate_params, template_matrix_norm, true_params, noise_voxel,
            )
            recovered_by_noise[noise_voxel] = recovered_params
            accuracy_by_noise[noise_voxel] = compute_accuracy_correlations(
                true_params, recovered_params, param_names=PARAM_NAMES)
            print(f"  Accuracy (true vs. per-voxel mean recovered Pearson r): "
                  f"{ {k: round(v, 3) for k, v in accuracy_by_noise[noise_voxel].items()} }")

            intermediate_path, scatter_plot_path, recovery_table_path = recovery_output_paths(
                design_seed, noise_voxel)
            np.savez(intermediate_path, true_params=true_params, recovered_params=recovered_params,
                     param_names=PARAM_NAMES)
            print(f"Recovery results saved -> {intermediate_path.resolve()}")
            plot_recovery(true_params, recovered_params, scatter_plot_path, param_names=PARAM_NAMES)
            save_recovery_table(true_params, recovered_params, recovery_table_path, param_names=PARAM_NAMES)

        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_plot_path(design_seed),
                               param_names=PARAM_NAMES)
        save_accuracy_table(accuracy_by_noise, accuracy_table_path(design_seed), param_names=PARAM_NAMES)
        plot_accuracy_comparison(accuracy_by_noise, accuracy_comparison_plot_path(design_seed),
                                  param_names=PARAM_NAMES, n_voxels=N_VOXELS, n_repeats=N_NOISE_REPEATS)


if __name__ == "__main__":
    main()
