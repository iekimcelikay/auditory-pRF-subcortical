"""kim2024_style_recovery_parallel_per_alpha.py
=================================================
Same Kim-et-al.-2024-inspired recovery procedure as
``kim2024_style_recovery.py`` (independent randomly-sampled voxels, Monte
Carlo noise repeats per voxel, post-hoc Pearson r accuracy reporting, single
shared ``tau_ms`` instead of separate ``tau_on_ms``/``tau_off_ms`` — see that
module's docstring for the full rationale) — this variant parallelizes the
candidate-grid build for dIPC.

Unlike an earlier version of this script, this candidate grid is NOT shared
with ``grid_recovery_parallel_per_alpha.py``'s ``build_candidate_grid_per_alpha``
— that builder loops over separate ``tau_on``/``tau_off`` grids and produces
5-column ``candidate_params``; once tau_on/tau_off were unified into one
``tau_ms`` parameter (per project discussion), the two candidate grids
genuinely diverged (4 columns, one fewer free dimension, ``tau_ms`` grid
shared for both ON and OFF), so this module has its own per-alpha-pool
builder below (same memory-bounding rationale — fixed 16-core/32GB dIPC
nodes — see ``grid_recovery_parallel_per_alpha.py``'s module docstring for
why a fresh ``Pool`` opens per alpha).

Usage (from project root, request multiple CPUs via SLURM — see
slurm_scripts/batch_kim2024_style_recovery_per_alpha.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_parallel_per_alpha

(``--no-capture-output`` matters — see kim2024_style_recovery.py's module
docstring: plain ``conda run`` withholds the child's stdout until it exits,
which would otherwise hide this script's per-alpha progress prints for the
whole candidate-grid build.)
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
from auditory_prf.prf_pipeline.solver.forward_model import (
    load_raw_sequences,
    sharpen_all_sequences,
    build_per_seq_trains,
    forward_bold_runs,
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
    N_DESIGN_SEEDS,
    DESIGN_SEEDS,
    NOISE_VOXELS,
    MC_SEED,
    N_NOISE_REPEATS,
    ALPHA_GRID,
    W_GRID,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    N_VOXELS,
    TAU_GRID_MS,
    PARAM_NAMES,
    RUN_DIR,
    candidate_grid_path,
    recovery_output_paths,
    noise_comparison_plot_path,
    accuracy_table_path,
    accuracy_comparison_plot_path,
    sample_ground_truths_off_grid,
    run_recovery,
    compute_accuracy_correlations,
    save_accuracy_table,
    plot_accuracy_comparison,
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
    """Pool initializer for ONE alpha's pool — sharpened data for that alpha only."""
    _worker["raw_sequences"] = raw_sequences
    _worker["sharpened_by_seq"] = sharpened_by_seq
    _worker["cf_list_hz"] = cf_list_hz
    _worker["run_designs"] = run_designs
    _worker["hrf_kernel"] = hrf_kernel
    _worker["per_seq_cache"] = {}   # cf_index -> (per_seq, cf_hz); bounded by n_cf


def _build_one_candidate(params: tuple) -> np.ndarray:
    """Worker task: forward-simulate one (cf_index, tau_ms, w) candidate for this alpha."""
    cf_index, tau_ms, w = params
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
        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S, w, tau_ms, tau_ms, RECTIFY, RHO,
    )
    return np.concatenate(bold_per_run).astype(np.float32)


def build_candidate_grid_per_alpha(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    n_workers: int = N_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-alpha-pool parallel version of ``kim2024_style_recovery.build_candidate_grid``.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4)
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(TAU_GRID_MS) * len(ALPHA_GRID) * len(W_GRID)
    print(f"Building candidate grid with {n_workers} worker processes (fresh pool per alpha): "
          f"{n_cf} CF x {len(TAU_GRID_MS)} tau x {len(ALPHA_GRID)} alpha x "
          f"{len(W_GRID)} w = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    chunk_size = len(TAU_GRID_MS) * len(W_GRID)   # one cf_index's worth
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)

        candidate_param_list = [
            (cf_index, tau_ms, w)
            for cf_index in range(n_cf)
            for tau_ms in TAU_GRID_MS
            for w in W_GRID
        ]

        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(raw_sequences, sharpened_by_seq, cf_list_hz, run_designs, hrf_kernel)) as pool:
            for local_i, template in enumerate(
                pool.imap(_build_one_candidate, candidate_param_list, chunksize=chunk_size)
            ):
                if template_matrix is None:
                    template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                cf_index, tau_ms, w = candidate_param_list[local_i]
                template_matrix[idx, :] = template
                candidate_params[idx] = (cf_index, tau_ms, alpha, w)
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
            raw_sequences, cf_list_hz, run_designs, hrf_kernel, n_workers=N_WORKERS,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths_off_grid(gt_rng, n_cf, N_VOXELS)

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning Kim-et-al.-style recovery at design_seed={design_seed}, "
                  f"noise_voxel='{noise_voxel}': {N_VOXELS} independent voxels, MC repeats each ...")
            recovered_params = run_recovery(
                raw_sequences, cf_list_hz, run_designs, hrf_kernel,
                candidate_params, template_matrix_norm, true_params, noise_voxel,
            )
            recovered_by_noise[noise_voxel] = recovered_params
            accuracy_by_noise[noise_voxel] = compute_accuracy_correlations(true_params, recovered_params)
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
        save_accuracy_table(accuracy_by_noise, accuracy_table_path(design_seed))
        plot_accuracy_comparison(accuracy_by_noise, accuracy_comparison_plot_path(design_seed),
                                  n_voxels=N_VOXELS, n_repeats=N_NOISE_REPEATS)


if __name__ == "__main__":
    main()
