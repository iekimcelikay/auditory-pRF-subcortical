"""grid_recovery_parallel_per_alpha.py
========================================
Same 5D grid-search parameter-recovery procedure as ``grid_recovery.py`` —
this variant parallelizes the candidate-grid build like
``grid_recovery_parallel_workers.py`` (a ``multiprocessing.Pool`` across the
AdapTrans+HRF forward-simulation step), but opens a **fresh Pool per alpha
value** instead of one Pool for the whole grid.

Why: with a single ``Pool.imap`` spanning all alphas
(``grid_recovery_parallel_workers.py``), worker task assignment is dynamic
and queue-based with no barrier between alpha blocks — a worker can end up
processing chunks from several different alphas over its lifetime, and since
its sharpened-population cache is never evicted, it can accumulate a cached
sharpened array (~120MB each) for every alpha it happens to touch, up to all
``len(ALPHA_GRID)`` of them. On a fixed-memory node (no option to just
request more RAM — see conversation: 16 cores x 32GB, non-negotiable), that
per-worker ceiling matters.

This version computes ``sharpen_all_sequences(alpha)`` ONCE in the main
process per alpha, passes that single alpha's already-sharpened data into a
Pool opened just for that alpha's candidates, then closes the Pool before
moving to the next alpha — bounding each worker's memory to exactly one
alpha's sharpened data (not up to all of them). Cost: 7 pool start/stops
instead of 1 (milliseconds each, negligible) and no cross-alpha work
stealing (a worker idle at the end of one alpha's candidates waits for the
next alpha's pool rather than grabbing another alpha's leftover chunk —
a minor efficiency loss, not a correctness issue).

Usage (from project root, request multiple CPUs via SLURM — see
slurm_scripts/batch_grid_recovery_per_alpha.slurm):
    conda run -n strf3.9 python -m auditory_prf.prf_pipeline.solver.grid_recovery_parallel_per_alpha
"""

import os
import time
from multiprocessing import Pool
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
)
from auditory_prf.prf_pipeline.solver.grid_recovery import (
    RESULTS_DIR,
    RUN_DIR,
    CANDIDATE_GRID_PATH,
    noise_comparison_plot_path,
    recovery_output_paths,
    TR_S,
    TOTAL_RUN_DUR_S,
    SIGNAL_DT_S,
    RECTIFY,
    RHO,
    CHUNK_MARGIN_MS,
    N_RUNS_RECOVERY,
    TAU_ON_GRID_MS,
    TAU_OFF_GRID_MS,
    ALPHA_GRID,
    W_GRID,
    N_GT_COMBOS,
    N_NOISE_REPEATS,
    NOISE_VOXELS,
    GT_SAMPLING_MODES,
    MC_SEED,
    PARAM_NAMES,
    sample_ground_truths,
    run_recovery,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)

# Number of worker processes for the candidate-grid build. Defaults to the
# SLURM-allocated CPU count when run under a SLURM job, else all local cores.
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# ── Worker-process globals (populated once per worker, per-alpha pool) ──────
_worker: dict = {}


def _init_worker(raw_sequences: dict, sharpened_by_seq: dict, cf_list_hz: np.ndarray,
                  run_designs: list, hrf_kernel: np.ndarray) -> None:
    """Pool initializer for ONE alpha's pool — sharpened data for that alpha only.

    Unlike grid_recovery_parallel_workers.py's ``_init_worker``, there is no
    per-alpha cache here: ``sharpened_by_seq`` already IS this alpha's data,
    fixed for this pool's entire lifetime (the pool is recreated per alpha).
    """
    _worker["raw_sequences"] = raw_sequences
    _worker["sharpened_by_seq"] = sharpened_by_seq
    _worker["cf_list_hz"] = cf_list_hz
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel
    _worker["per_seq_cache"] = {}   # cf_index -> (per_seq, cf_hz); bounded by n_cf, small either way


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, tau_on, tau_off, w) candidate for this alpha."""
    cf_index, tau_on_ms, tau_off_ms, w = params
    cf_index = int(cf_index)

    per_seq_cache = _worker["per_seq_cache"]
    if cf_index not in per_seq_cache:
        per_seq_cache[cf_index] = build_per_seq_trains(
            _worker["raw_sequences"], _worker["sharpened_by_seq"], cf_index, _worker["cf_list_hz"],
            TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
        )
    per_seq, cf_hz = per_seq_cache[cf_index]

    bold_per_run = forward_bold_runs(
        per_seq, _worker["run_designs"], _worker["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, w, tau_on_ms, tau_off_ms, RECTIFY, RHO,
    )
    return np.concatenate(bold_per_run).astype(np.float32)


def build_candidate_grid_per_alpha(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of ``grid_recovery.build_candidate_grid`` — same output.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 5)
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(TAU_ON_GRID_MS) * len(TAU_OFF_GRID_MS) * len(ALPHA_GRID) * len(W_GRID)
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha): "
          f"{n_cf} CF x {len(TAU_ON_GRID_MS)} tau_on x {len(TAU_OFF_GRID_MS)} tau_off x "
          f"{len(ALPHA_GRID)} alpha x {len(W_GRID)} w = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 5), dtype=float)
    template_matrix = None
    chunk_size = len(TAU_ON_GRID_MS) * len(TAU_OFF_GRID_MS) * len(W_GRID)   # one cf_index's worth
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        # Sharpened once here, in the main process — passed into this
        # alpha's pool via initargs, never recomputed or cached per-worker.
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)

        candidate_param_list = [
            (cf_index, tau_on_ms, tau_off_ms, w)
            for cf_index in range(n_cf)
            for tau_on_ms in TAU_ON_GRID_MS
            for tau_off_ms in TAU_OFF_GRID_MS
            for w in W_GRID
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, sharpened_by_seq, cf_list_hz, run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, tau_on_ms, tau_off_ms, w = candidate_param_list[local_i]
                template_matrix[idx, :] = template
                candidate_params[idx] = (cf_index, tau_on_ms, tau_off_ms, alpha, w)
                idx += 1
        # Pool (and every worker's memory, including sharpened_by_seq) is
        # fully released here, before the next alpha's pool opens.

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

    candidate_params, template_matrix_norm = build_candidate_grid_per_alpha(
        raw_sequences, cf_list_hz, run_designs, hrf_kernel, n_workers=N_WORKERS,
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
            plot_recovery(true_params, recovered_params, scatter_plot_path)
            save_recovery_table(true_params, recovered_params, recovery_table_path)

        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_plot_path(gt_mode))


if __name__ == "__main__":
    main()
