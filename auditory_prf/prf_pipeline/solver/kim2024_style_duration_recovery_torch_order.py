"""kim2024_style_duration_recovery_torch_order.py
===================================================
Same Kim-et-al.-2024-inspired duration-only recovery procedure as
``kim2024_style_duration_recovery.py``, but builds candidates with
``forward_model_torch_order_fixed_reference.build_per_seq_trains_torch_order_fixed_reference``
instead of ``duration_recovery.build_candidate_grid`` (which uses
``forward_model.build_per_seq_trains``'s order) — see
``kim2024_style_recovery_torch_order.py``'s module docstring for the full
chain of reasoning (stage order + fixed-reference calibration).

``forward_bold_runs`` with ``apply_adaptrans_flag=False`` is reused
UNCHANGED from forward_model.py, same as every other duration-only variant
in this project.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order
"""

import time
from datetime import datetime
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
    forward_bold_runs,
    draw_noisy_measurement,
    spawn_noise_seeds,
)
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
    ALPHA_GRID,
)
from auditory_prf.prf_pipeline.solver.duration_recovery import (
    PARAM_NAMES,
    N_NOISE_REPEATS,
    PREF_DUR_GRID_MS,
    SIGMA_DUR_GRID_MS,
    sample_ground_truths_off_grid,
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

# ── Paths (own base dir/timestamp) ──────────────────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR    = Path("kim2024_style_duration_recovery_torch_order_results")
RUN_DIR       = OUTPUT_DIR / RUN_TIMESTAMP


def candidate_grid_path(design_seed: int) -> Path:
    return RUN_DIR / f"candidate_grid_design{design_seed}_{RUN_TIMESTAMP}.npz"


def recovery_output_paths(design_seed: int, noise_voxel: str) -> tuple[Path, Path, Path]:
    tag = f"design{design_seed}_{noise_voxel}_{RUN_TIMESTAMP}"
    return (
        RUN_DIR / f"kim_duration_recovery_results_{tag}.npz",
        RUN_DIR / f"kim_duration_recovery_scatter_{tag}.png",
        RUN_DIR / f"kim_duration_recovery_table_{tag}.csv",
    )


def noise_comparison_plot_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_duration_recovery_noise_comparison_design{design_seed}_{RUN_TIMESTAMP}.png"


def accuracy_table_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_duration_recovery_accuracy_table_design{design_seed}_{RUN_TIMESTAMP}.csv"


def accuracy_comparison_plot_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_duration_recovery_accuracy_comparison_design{design_seed}_{RUN_TIMESTAMP}.png"


def build_candidate_grid(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    reference_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate every (cf_index, pref_dur_ms, sigma_dur_ms, alpha)
    candidate, torch order + fixed reference."""
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(PREF_DUR_GRID_MS) * len(SIGMA_DUR_GRID_MS) * len(ALPHA_GRID)
    print(f"Building candidate grid (torch order): {n_cf} CF x {len(PREF_DUR_GRID_MS)} pref_dur x "
          f"{len(SIGMA_DUR_GRID_MS)} sigma_dur x {len(ALPHA_GRID)} alpha = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        for cf_index in range(n_cf):
            for pref_dur_ms in PREF_DUR_GRID_MS:
                for sigma_dur_ms in SIGMA_DUR_GRID_MS:
                    # Unlike grid_recovery.py's order, per_seq must be
                    # rebuilt for every (cf_index, pref_dur_ms, sigma_dur_ms,
                    # alpha) combination here too -- no caching benefit left
                    # (same as duration_recovery.py's own note, now also
                    # true at fixed alpha since sharpening depends on
                    # cf_index jointly in this stage order).
                    per_seq, cf_hz = build_per_seq_trains_torch_order_fixed_reference(
                        raw_sequences, cf_index, cf_list_hz, alpha, TC_SILENCE_SEQ_ID,
                        reference_table, chunk_margin_ms=CHUNK_MARGIN_MS,
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


def run_recovery(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    reference_table: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise_voxel: str,
) -> np.ndarray:
    """Monte Carlo recovery loop for one noise level (torch order + fixed reference)."""
    n_voxels = true_params.shape[0]
    recovered_params = np.empty((n_voxels, N_NOISE_REPEATS, 4), dtype=float)

    for v_i, (cf_index_f, pref_dur_ms, sigma_dur_ms, alpha) in enumerate(true_params):
        cf_index = int(round(cf_index_f))
        per_seq, cf_hz = build_per_seq_trains_torch_order_fixed_reference(
            raw_sequences, cf_index, cf_list_hz, alpha, TC_SILENCE_SEQ_ID,
            reference_table, chunk_margin_ms=CHUNK_MARGIN_MS,
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
            run_seeds = spawn_noise_seeds(MC_SEED, v_i, rep, n_runs=N_RUNS_RECOVERY)
            measurements[rep, :] = draw_noisy_measurement(
                gt_bold_per_run, noise_voxel, TR_S, run_seeds,
            )

        measurements -= measurements.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(measurements, axis=1, keepdims=True)
        measurements /= np.where(norms == 0, 1.0, norms)

        r = template_matrix_norm @ measurements.T
        best_idx = np.argmax(r, axis=0)
        recovered_params[v_i, :, :] = candidate_params[best_idx]

        bias = recovered_params[v_i].mean(axis=0) - [cf_index, pref_dur_ms, sigma_dur_ms, alpha]
        print(f"  [{noise_voxel}] voxel {v_i+1}/{n_voxels}: true cf_index={cf_index} "
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

        candidate_params, template_matrix_norm = build_candidate_grid(
            raw_sequences, cf_list_hz, run_designs, hrf_kernel, reference_table,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths_off_grid(gt_rng, n_cf, N_VOXELS)

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
