"""kim2024_style_duration_recovery_torch_order_parallel_per_alpha.py
======================================================================
Same duration-only recovery procedure as
``kim2024_style_duration_recovery_torch_order.py`` (torch_pipeline.py's
stage order + fixed per-CF reference), this variant parallelizes the
candidate-grid build with a fresh ``multiprocessing.Pool`` per alpha.

No per-worker caching benefit here (same as duration_recovery_parallel_per_alpha.py's
own note): ``per_seq`` depends on (cf_index, alpha, pref_dur_ms,
sigma_dur_ms) jointly in this stage order, so every candidate rebuilds it
from scratch regardless of worker reuse.

Usage (from project root, request multiple CPUs via SLURM — see
slurm_scripts/batch_kim2024_style_duration_recovery_torch_order_per_alpha.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha
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
    ALPHA_GRID,
)
from auditory_prf.prf_pipeline.solver.duration_recovery import (
    PARAM_NAMES,
    N_NOISE_REPEATS,
    PREF_DUR_GRID_MS,
    n_scaled_sigma_candidates,
    sigma_grid_for_pref_dur,
    draw_scaled_ground_truths,
    _UNUSED_W,
    _UNUSED_TAU_MS,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    save_accuracy_table,
    plot_accuracy_comparison,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery import N_VOXELS
# Ground-truth sampling: "on_grid" = random candidate-grid rows (noise-free recovery should be exact),
# "off_grid" = continuous draws (also exposes grid-quantization error).
GT_MODE = "off_grid"
from auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order import (
    RUN_DIR,
    candidate_grid_path,
    recovery_output_paths,
    noise_comparison_plot_path,
    accuracy_table_path,
    accuracy_comparison_plot_path,
)
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# SLURM array: task i runs design seed DESIGN_SEEDS[i] only (see the .slurm's --array).
if "SLURM_ARRAY_TASK_ID" in os.environ:
    DESIGN_SEEDS = [DESIGN_SEEDS[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    N_DESIGN_SEEDS = len(DESIGN_SEEDS)

_worker: dict = {}


def _init_worker(raw_sequences: dict, cf_list_hz: np.ndarray, alpha: float,
                  population_pool: np.ndarray, run_designs: list, hrf_kernel: np.ndarray) -> None:
    _worker["raw_sequences"] = raw_sequences
    _worker["cf_list_hz"] = cf_list_hz
    _worker["alpha"] = alpha
    _worker["population_pool"] = population_pool
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, pref_dur_ms, sigma_dur_ms) candidate for this alpha."""
    cf_index, pref_dur_ms, sigma_dur_ms = params
    cf_index = int(cf_index)

    per_seq, cf_hz = build_per_seq_trains_torch_order_fixed_reference(
        _worker["raw_sequences"], cf_index, _worker["cf_list_hz"], _worker["alpha"],
        TC_SILENCE_SEQ_ID, _worker["population_pool"], chunk_margin_ms=CHUNK_MARGIN_MS,
        pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
    )
    bold_per_run = forward_bold_runs(
        per_seq, _worker["run_designs"], _worker["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, _UNUSED_W, _UNUSED_TAU_MS, _UNUSED_TAU_MS, RECTIFY, RHO,
        apply_adaptrans_flag=False,
    )
    return np.concatenate(bold_per_run).astype(np.float32)


def build_candidate_grid_per_alpha(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    population_pool: np.ndarray,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of
    ``kim2024_style_duration_recovery_torch_order.build_candidate_grid``."""
    n_cf = len(cf_list_hz)
    n_pref_sigma_pairs = n_scaled_sigma_candidates()
    n_candidates = n_cf * n_pref_sigma_pairs * len(ALPHA_GRID)
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha, torch order): "
          f"{n_cf} CF x {n_pref_sigma_pairs} (pref_dur, scaled sigma_dur) pairs x "
          f"{len(ALPHA_GRID)} alpha = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    chunk_size = n_pref_sigma_pairs
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        candidate_param_list = [
            (cf_index, pref_dur_ms, sigma_dur_ms)
            for cf_index in range(n_cf)
            for pref_dur_ms in PREF_DUR_GRID_MS
            for sigma_dur_ms in sigma_grid_for_pref_dur(pref_dur_ms)
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, cf_list_hz, alpha, population_pool, run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, pref_dur_ms, sigma_dur_ms = candidate_param_list[local_i]
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


def _recover_one_voxel(v_i: int, true_row: np.ndarray, state: dict) -> np.ndarray:
    """Worker task: all MC noise repeats for one ground-truth voxel.

    Returns
    -------
    np.ndarray, shape (N_NOISE_REPEATS, 4)
        Recovered (cf_index, pref_dur_ms, sigma_dur_ms, alpha) per repeat.
    """
    cf_index_f, pref_dur_ms, sigma_dur_ms, alpha = true_row
    cf_index = int(round(cf_index_f))
    per_seq, cf_hz = build_per_seq_trains_torch_order_fixed_reference(
        state["raw_sequences"], cf_index, state["cf_list_hz"], alpha, TC_SILENCE_SEQ_ID,
        state["population_pool"], chunk_margin_ms=CHUNK_MARGIN_MS,
        pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
    )
    gt_bold_per_run = forward_bold_runs(
        per_seq, state["run_designs"], state["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, _UNUSED_W, _UNUSED_TAU_MS, _UNUSED_TAU_MS, RECTIFY, RHO,
        apply_adaptrans_flag=False,
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
    ``kim2024_style_duration_recovery_torch_order.run_recovery`` (same seeds,
    same per-voxel procedure; repeats are scored in memory-bounded chunks).

    Returns
    -------
    np.ndarray, shape (n_voxels, N_NOISE_REPEATS, 4)
    """
    n_voxels = true_params.shape[0]
    recovered_params = np.empty((n_voxels, N_NOISE_REPEATS, 4), dtype=float)
    state = dict(
        raw_sequences=raw_sequences, cf_list_hz=cf_list_hz, run_designs=run_designs,
        hrf_kernel=hrf_kernel, population_pool=population_pool,
        candidate_params=candidate_params, template_matrix_norm=template_matrix_norm,
        noise_voxel=noise_voxel,
    )
    for v_i, voxel_recovered in enumerate(map_voxels(_recover_one_voxel, true_params, state, n_workers)):
        recovered_params[v_i] = voxel_recovered
        cf_index_f, pref_dur_ms, sigma_dur_ms, alpha = true_params[v_i]
        bias = voxel_recovered.mean(axis=0) - [int(round(cf_index_f)), pref_dur_ms, sigma_dur_ms, alpha]
        print(f"  [{noise_voxel}] voxel {v_i+1}/{n_voxels}: true cf_index={int(round(cf_index_f))} "
              f"pref_dur={pref_dur_ms:.1f} sigma_dur={sigma_dur_ms:.1f} alpha={alpha:.3f} "
              f"| bias={np.round(bias, 3)}", flush=True)
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
            raw_sequences, cf_list_hz, run_designs, hrf_kernel, population_pool, n_workers=N_WORKERS,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = draw_scaled_ground_truths(GT_MODE, gt_rng, n_cf, N_VOXELS, candidate_params)

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning Kim-et-al.-style duration recovery at design_seed={design_seed}, "
                  f"noise_voxel='{noise_voxel}': {N_VOXELS} independent voxels, MC repeats each ...")
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
