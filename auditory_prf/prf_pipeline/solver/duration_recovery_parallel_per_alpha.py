"""duration_recovery_parallel_per_alpha.py
===========================================
Same duration-model-only (AdapTrans off) parameter-recovery procedure as
``duration_recovery.py`` -- this variant parallelizes the candidate-grid
build with a fresh ``multiprocessing.Pool`` per alpha value, same rationale
and memory-bounding as ``grid_recovery_parallel_per_alpha.py`` (fixed
16-core/32GB nodes, no option to just request more RAM).

Simpler than the AdapTrans version's worker: there, ``build_per_seq_trains``
only depended on (alpha, cf_index), so workers cached it across many
candidates sharing a cf_index. Here, the duration-Gaussian filter runs
*inside* ``build_per_seq_trains``, so per_seq must be rebuilt for every
(cf_index, pref_dur_ms, sigma_dur_ms) triple within one alpha's pool --
there is nothing left to cache per worker (see duration_recovery.py's
module docstring).

Usage (from project root, request multiple CPUs via SLURM -- see
slurm_scripts/batch_duration_recovery_per_alpha.slurm):
    conda run -n strf3.9 python -m auditory_prf.prf_pipeline.solver.duration_recovery_parallel_per_alpha
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
from auditory_prf.prf_pipeline.solver.duration_recovery import (
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
    PREF_DUR_GRID_MS,
    SIGMA_DUR_GRID_MS,
    ALPHA_GRID,
    N_GT_COMBOS,
    N_NOISE_REPEATS,
    NOISE_VOXELS,
    GT_SAMPLING_MODES,
    MC_SEED,
    PARAM_NAMES,
    _UNUSED_W,
    _UNUSED_TAU_MS,
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
    """Pool initializer for ONE alpha's pool -- sharpened data for that alpha only."""
    _worker["raw_sequences"] = raw_sequences
    _worker["sharpened_by_seq"] = sharpened_by_seq
    _worker["cf_list_hz"] = cf_list_hz
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, pref_dur_ms, sigma_dur_ms) candidate for this alpha.

    No per-worker caching here (unlike the AdapTrans version) -- per_seq
    depends on the duration params too, so every candidate needs its own
    ``build_per_seq_trains`` call regardless.
    """
    cf_index, pref_dur_ms, sigma_dur_ms = params
    cf_index = int(cf_index)

    per_seq, cf_hz = build_per_seq_trains(
        _worker["raw_sequences"], _worker["sharpened_by_seq"], cf_index, _worker["cf_list_hz"],
        TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
        pref_dur_ms=pref_dur_ms, sigma_dur_ms=sigma_dur_ms,
    )
    bold_per_run = forward_bold_runs(
        per_seq, _worker["run_designs"], _worker["hrf_kernel"], cf_hz,
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
        _UNUSED_W, _UNUSED_TAU_MS, _UNUSED_TAU_MS, RECTIFY, RHO,
        apply_adaptrans_flag=False,
    )
    return np.concatenate(bold_per_run).astype(np.float32)


def build_candidate_grid_per_alpha(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of ``duration_recovery.build_candidate_grid`` -- same output.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4)
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(PREF_DUR_GRID_MS) * len(SIGMA_DUR_GRID_MS) * len(ALPHA_GRID)
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha): "
          f"{n_cf} CF x {len(PREF_DUR_GRID_MS)} pref_dur x {len(SIGMA_DUR_GRID_MS)} sigma_dur x "
          f"{len(ALPHA_GRID)} alpha = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    chunk_size = len(PREF_DUR_GRID_MS) * len(SIGMA_DUR_GRID_MS)   # one cf_index's worth
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)

        candidate_param_list = [
            (cf_index, pref_dur_ms, sigma_dur_ms)
            for cf_index in range(n_cf)
            for pref_dur_ms in PREF_DUR_GRID_MS
            for sigma_dur_ms in SIGMA_DUR_GRID_MS
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, sharpened_by_seq, cf_list_hz, run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, pref_dur_ms, sigma_dur_ms = candidate_param_list[local_i]
                template_matrix[idx, :] = template
                candidate_params[idx] = (cf_index, pref_dur_ms, sigma_dur_ms, alpha)
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
            plot_recovery(true_params, recovered_params, scatter_plot_path, param_names=PARAM_NAMES)
            save_recovery_table(true_params, recovered_params, recovery_table_path, param_names=PARAM_NAMES)

        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_plot_path(gt_mode),
                               param_names=PARAM_NAMES)


if __name__ == "__main__":
    main()
