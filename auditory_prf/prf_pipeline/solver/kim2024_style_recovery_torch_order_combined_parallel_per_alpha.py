"""kim2024_style_recovery_torch_order_combined_parallel_per_alpha.py
======================================================================
Same combined AdapTrans + duration-selectivity recovery procedure as
``kim2024_style_recovery_torch_order_combined.py``, this variant
parallelizes the candidate-grid build with a fresh ``multiprocessing.Pool``
per alpha, same rationale as ``grid_recovery_parallel_per_alpha.py``.

Per-worker caching: ``per_seq`` depends on (cf_index, alpha, pref_dur_ms,
sigma_dur_ms) -- NOT on tau_ms/w, which only affect the later AdapTrans+HRF
stage -- so each worker caches ``per_seq`` keyed by (cf_index, pref_dur_ms,
sigma_dur_ms) within one alpha's pool, reused across every (k_tau, w)
combination. ``cf_to_tau_ms`` is cheap pure arithmetic, so it's recomputed
in both the main process (to record candidate_params) and inside each
worker (to build the actual candidate) rather than threading it through
the task tuple.

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
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import cf_to_tau_ms
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
from auditory_prf.prf_pipeline.solver.reference_calibration import build_per_cf_reference_table
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
    K_TAU_GRID,
    W_GRID,
    TAU_RANGE_MS,
    RUN_DIR,
    candidate_grid_path,
    recovery_output_paths,
    noise_comparison_plot_path,
    accuracy_table_path,
    accuracy_comparison_plot_path,
    sample_ground_truths_off_grid,
    run_recovery,
)

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

_worker: dict = {}


def _init_worker(raw_sequences: dict, cf_list_hz: np.ndarray, alpha: float,
                  reference_table: np.ndarray, cf_range_hz: tuple,
                  run_designs: list, hrf_kernel: np.ndarray) -> None:
    _worker["raw_sequences"] = raw_sequences
    _worker["cf_list_hz"] = cf_list_hz
    _worker["alpha"] = alpha
    _worker["reference_table"] = reference_table
    _worker["cf_range_hz"] = cf_range_hz
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel
    _worker["per_seq_cache"] = {}


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, pref_dur_ms, sigma_dur_ms, k_tau, w) candidate."""
    cf_index, pref_dur_ms, sigma_dur_ms, k_tau, w = params
    cf_index = int(cf_index)

    per_seq_cache = _worker["per_seq_cache"]
    cache_key = (cf_index, pref_dur_ms, sigma_dur_ms)
    if cache_key not in per_seq_cache:
        per_seq_cache[cache_key] = build_per_seq_trains_torch_order_fixed_reference(
            _worker["raw_sequences"], cf_index, _worker["cf_list_hz"], _worker["alpha"],
            TC_SILENCE_SEQ_ID, _worker["reference_table"], chunk_margin_ms=CHUNK_MARGIN_MS,
            pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
        )
    per_seq, cf_hz = per_seq_cache[cache_key]

    tau_anchor_ms = cf_to_tau_ms(cf_hz, _worker["cf_range_hz"], TAU_RANGE_MS)
    tau_ms = k_tau * tau_anchor_ms

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
    reference_table: np.ndarray,
    cf_range_hz: tuple,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of
    ``kim2024_style_recovery_torch_order_combined.build_candidate_grid``."""
    n_cf = len(cf_list_hz)
    n_candidates = (n_cf * len(PREF_DUR_GRID_MS) * len(K_SIGMA_GRID)
                    * len(ALPHA_GRID) * len(K_TAU_GRID) * len(W_GRID))
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha, "
          f"torch order, combined AdapTrans+duration): {n_cf} CF x {len(PREF_DUR_GRID_MS)} pref_dur x "
          f"{len(K_SIGMA_GRID)} k_sigma x {len(ALPHA_GRID)} alpha x {len(K_TAU_GRID)} k_tau x "
          f"{len(W_GRID)} w = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 6), dtype=float)
    template_matrix = None
    chunk_size = len(K_TAU_GRID) * len(W_GRID)   # one (cf_index, pref_dur, k_sigma)'s worth
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        candidate_param_list = [
            (cf_index, pref_dur_ms, k_sigma * pref_dur_ms, k_tau, w)
            for cf_index in range(n_cf)
            for pref_dur_ms in PREF_DUR_GRID_MS
            for k_sigma in K_SIGMA_GRID
            for k_tau in K_TAU_GRID
            for w in W_GRID
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, cf_list_hz, alpha, reference_table, cf_range_hz,
                            run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, pref_dur_ms, sigma_dur_ms, k_tau, w = candidate_param_list[local_i]
                cf_hz = float(cf_list_hz[int(cf_index)])
                tau_anchor_ms = cf_to_tau_ms(cf_hz, cf_range_hz, TAU_RANGE_MS)
                tau_ms = k_tau * tau_anchor_ms
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

    print("Building per-CF fixed reference table (pooled across active sequences) ...")
    reference_table = build_per_cf_reference_table(
        raw_sequences, cf_list_hz, TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
    )
    print(f"  reference_table shape: {reference_table.shape}")

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
            raw_sequences, cf_list_hz, run_designs, hrf_kernel, reference_table, cf_range_hz,
            n_workers=N_WORKERS,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths_off_grid(gt_rng, n_cf, cf_list_hz, cf_range_hz, N_VOXELS)

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning recovery at design_seed={design_seed}, noise_voxel='{noise_voxel}': "
                  f"{N_VOXELS} independent voxels x {N_NOISE_REPEATS:,} MC noise repeats each ...")
            recovered_params = run_recovery(
                raw_sequences, cf_list_hz, run_designs, hrf_kernel, reference_table,
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
