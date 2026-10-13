"""kim2024_style_recovery.py
=============================
AdapTrans parameter-recovery test restructured to mirror Kim et al. 2024's
("Characterizing Spatiotemporal Population Receptive Fields in Human Visual
Cortex with fMRI") simulation/validation design, as closely as it applies to
this project's own (unrelated) subcortical AdapTrans+duration model. This is
NOT a GLM/SPM-style script — per project discussion, reading that paper's
Methods in full turned up no GLM, ReML, AR(1), or whitening anywhere in its
simulation/Solver sections (SPM12 appears exactly once, for real-data motion
correction only). What the paper's "300 simulated voxels" design actually is:

  - N independent, randomly (continuously, i.e. off-grid) sampled
    ground-truth parameter sets — NOT a handful of ground truths each hit
    with thousands of Monte Carlo noise repeats (contrast with
    ``grid_recovery.py``'s ``N_GT_COMBOS=20`` x ``N_NOISE_REPEATS=10_000``).
  - Each ground truth gets exactly ONE noisy synthetic timecourse.
  - Accuracy is reported post-hoc (their Fig. 4E): Pearson r between the N
    true values and the N recovered values, across voxels, per parameter —
    a fundamentally different use of "Pearson r" than the grid search's own
    fitting/argmax criterion.

Per project discussion, two of the paper's choices are deliberately NOT
replicated here (both would be larger, separate changes):
  - Fitting stays grid-search argmax over Pearson r (reusing
    ``grid_recovery.build_candidate_grid`` as-is), not BADS continuous
    refinement.
  - The fitting criterion stays Pearson r argmax (amplitude-invariant), not
    the paper's raw-timecourse R² (Eq. 12) — switching would require
    deciding how candidate template amplitude/gain is handled, which the
    model has no free parameter for today.

Also deliberately NOT replicated: the paper's single fixed-SNR noise
calibration — this script keeps this project's existing low/mid/high
NOISE_VOXELS sweep instead (per project discussion), and its design-seed
sweep (grid_recovery.py's DESIGN_SEEDS, reused here) — the paper has no
analogous multi-run-ordering concept, but this project already checks
whether findings hold across trial orderings, so that axis is kept.

GT sampling is off-grid only (continuous, via
``grid_recovery.sample_ground_truths_off_grid``) — the natural match for the
paper's "randomly sampled within bounds" design; there is no on-grid mode
here (unlike grid_recovery.py's GT_SAMPLING_MODES sweep).

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery

(``--no-capture-output`` matters: plain ``conda run`` buffers/withholds the
child process's stdout until it exits — even with ``python -u`` — so without
it, this script's per-alpha/per-voxel progress prints stay invisible for the
whole candidate-grid build, which can look exactly like a hang. See project
discussion.)
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    N_DESIGN_SEEDS,
    DESIGN_SEEDS,
    MC_SEED,
    PARAM_NAMES,
    build_candidate_grid,
    sample_ground_truths_off_grid,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)

# ── This script's own sweep size — independent random voxels, one noisy
# draw each (see module docstring). First-pass value per project discussion
# (paper uses 300; start smaller while shaking out this new sampling/
# accuracy-reporting scheme) — bump later once this runs cleanly on dIPC. ──
N_VOXELS = 100

# ── Paths (separate from grid_recovery.py's, own base dir/timestamp) ───────
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR    = Path("kim2024_style_recovery_results")
RUN_DIR       = OUTPUT_DIR / RUN_TIMESTAMP


def candidate_grid_path(design_seed: int) -> Path:
    return RUN_DIR / f"candidate_grid_design{design_seed}_{RUN_TIMESTAMP}.npz"


def recovery_output_paths(design_seed: int, noise_voxel: str) -> tuple[Path, Path, Path]:
    tag = f"design{design_seed}_{noise_voxel}_{RUN_TIMESTAMP}"
    return (
        RUN_DIR / f"kim_recovery_results_{tag}.npz",
        RUN_DIR / f"kim_recovery_scatter_{tag}.png",
        RUN_DIR / f"kim_recovery_table_{tag}.csv",
    )


def noise_comparison_plot_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_recovery_noise_comparison_design{design_seed}_{RUN_TIMESTAMP}.png"


def accuracy_table_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_recovery_accuracy_table_design{design_seed}_{RUN_TIMESTAMP}.csv"


def accuracy_comparison_plot_path(design_seed: int) -> Path:
    return RUN_DIR / f"kim_recovery_accuracy_comparison_design{design_seed}_{RUN_TIMESTAMP}.png"


def run_recovery_single_draw(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise_voxel: str,
) -> np.ndarray:
    """One independent ground truth -> one noisy measurement -> one recovered
    fit, for every voxel in ``true_params`` (Kim et al. 2024's "simulated
    voxels" design — contrast with ``grid_recovery.run_recovery``'s handful
    of ground truths x thousands of Monte Carlo repeats each).

    Noise seed is keyed on ``v_i`` alone (not ``noise_voxel`` or a repeat
    index) — same trick as ``grid_recovery.run_recovery``: identical noise
    *pattern* reused across noise levels, only the ``PmNoise`` amplitude
    preset differs, so any difference across levels is attributable to noise
    amplitude alone.

    Returns
    -------
    recovered_params : np.ndarray, shape (n_voxels, 5)
    """
    n_voxels = true_params.shape[0]
    recovered_params = np.empty((n_voxels, 5), dtype=float)

    for v_i, (cf_index_f, tau_on_ms, tau_off_ms, alpha, w) in enumerate(true_params):
        cf_index = int(round(cf_index_f))
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        per_seq, cf_hz = build_per_seq_trains(
            raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
            TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
        )
        gt_bold_per_run = forward_bold_runs(
            per_seq, run_designs, hrf_kernel, cf_hz,
            TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
            w, tau_on_ms, tau_off_ms, RECTIFY, RHO,
        )

        run_seeds = spawn_noise_seeds(MC_SEED, v_i, n_runs=N_RUNS_RECOVERY)
        measurement = draw_noisy_measurement(gt_bold_per_run, noise_voxel, TR_S, run_seeds)

        measurement = measurement - measurement.mean()
        norm = np.linalg.norm(measurement)
        if norm != 0:
            measurement = measurement / norm

        r = template_matrix_norm @ measurement
        best_idx = np.argmax(r)
        recovered_params[v_i, :] = candidate_params[best_idx]

        print(f"  [{noise_voxel}] voxel {v_i+1}/{n_voxels}: true cf_index={cf_index} "
              f"tau_on={tau_on_ms:.1f} tau_off={tau_off_ms:.1f} alpha={alpha:.3f} w={w:.3f} "
              f"-> recovered {np.round(recovered_params[v_i], 3)}")

    return recovered_params


def compute_accuracy_correlations(
    true_params: np.ndarray, recovered_params: np.ndarray, param_names: list = PARAM_NAMES,
) -> dict:
    """Post-hoc accuracy statistic, Kim et al. 2024 Fig. 4E style: Pearson r
    between true and recovered values across all independent voxels, one r
    per parameter.

    This is NOT the grid search's own fitting criterion (that stays Pearson
    r argmax against the candidate grid, computed once per voxel inside
    ``run_recovery_single_draw``) — this is a separate, after-the-fact
    summary of how well recovery tracked ground truth across voxels.
    """
    return {
        name: float(np.corrcoef(true_params[:, p], recovered_params[:, p])[0, 1])
        for p, name in enumerate(param_names)
    }


def save_accuracy_table(accuracy_by_noise: dict, save_path: Path, param_names: list = PARAM_NAMES) -> None:
    """One row per noise level, one column per parameter's accuracy r."""
    header = ["noise_voxel"] + [f"r_{name}" for name in param_names]
    with open(save_path, "w") as f:
        f.write(",".join(header) + "\n")
        for noise_voxel, accuracy in accuracy_by_noise.items():
            row = [noise_voxel] + [f"{accuracy[name]:.4f}" for name in param_names]
            f.write(",".join(row) + "\n")
    print(f"Accuracy-correlation table saved -> {save_path.resolve()}")


def plot_accuracy_comparison(accuracy_by_noise: dict, save_path: Path, param_names: list = PARAM_NAMES) -> None:
    """Bar chart of the post-hoc accuracy statistic: one group per parameter,
    one bar per noise level, height = Pearson r between true and recovered
    values across all ``N_VOXELS`` independent draws (Fig. 4E style)."""
    level_colors = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
    noise_levels = list(accuracy_by_noise.keys())
    n_params = len(param_names)
    x = np.arange(n_params)
    width = 0.8 / max(len(noise_levels), 1)

    fig, ax = plt.subplots(figsize=(1.5 * n_params + 2, 4))
    for i, noise_voxel in enumerate(noise_levels):
        heights = [accuracy_by_noise[noise_voxel][name] for name in param_names]
        ax.bar(x + i * width, heights, width, label=noise_voxel,
               color=level_colors.get(noise_voxel, "steelblue"))

    ax.set_xticks(x + width * (len(noise_levels) - 1) / 2)
    ax.set_xticklabels(param_names)
    ax.set_ylabel("Pearson r (true vs. recovered, across voxels)")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.legend(title="noise level")
    ax.set_title(f"Recovery accuracy across noise levels (N={N_VOXELS} voxels)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy comparison plot saved -> {save_path.resolve()}")


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

        candidate_params, template_matrix_norm = build_candidate_grid(
            raw_sequences, cf_list_hz, run_designs, hrf_kernel,
        )
        cgp = candidate_grid_path(design_seed)
        np.savez(cgp, candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)
        print(f"Candidate grid saved -> {cgp.resolve()}")

        # One set of N_VOXELS independent off-grid ground truths per design
        # seed (same MC_SEED every design seed, like grid_recovery.py, so the
        # ground truths themselves don't confound a design-seed comparison).
        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths_off_grid(gt_rng, n_cf, N_VOXELS)

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning Kim-et-al.-style recovery at design_seed={design_seed}, "
                  f"noise_voxel='{noise_voxel}': {N_VOXELS} independent voxels, "
                  f"one noisy draw each ...")
            recovered_params = run_recovery_single_draw(
                raw_sequences, cf_list_hz, run_designs, hrf_kernel,
                candidate_params, template_matrix_norm, true_params, noise_voxel,
            )
            recovered_by_noise[noise_voxel] = recovered_params
            accuracy_by_noise[noise_voxel] = compute_accuracy_correlations(true_params, recovered_params)
            print(f"  Accuracy (true vs. recovered Pearson r): "
                  f"{ {k: round(v, 3) for k, v in accuracy_by_noise[noise_voxel].items()} }")

            intermediate_path, scatter_plot_path, recovery_table_path = recovery_output_paths(
                design_seed, noise_voxel)
            np.savez(intermediate_path, true_params=true_params, recovered_params=recovered_params,
                     param_names=PARAM_NAMES)
            print(f"Recovery results saved -> {intermediate_path.resolve()}")

            # plot_recovery/save_recovery_table expect a (n_gt, n_repeats,
            # n_params) recovered array -- add a length-1 "repeats" axis to
            # reuse them unmodified for this one-draw-per-voxel design.
            recovered_3d = recovered_params[:, None, :]
            plot_recovery(true_params, recovered_3d, scatter_plot_path)
            save_recovery_table(true_params, recovered_3d, recovery_table_path)

        recovered_by_noise_3d = {k: v[:, None, :] for k, v in recovered_by_noise.items()}
        plot_noise_comparison(true_params, recovered_by_noise_3d, noise_comparison_plot_path(design_seed))
        save_accuracy_table(accuracy_by_noise, accuracy_table_path(design_seed))
        plot_accuracy_comparison(accuracy_by_noise, accuracy_comparison_plot_path(design_seed))


if __name__ == "__main__":
    main()
