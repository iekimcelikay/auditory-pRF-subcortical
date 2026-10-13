"""kim2024_style_recovery.py
=============================
AdapTrans parameter-recovery test, restructured per project discussion in
two ways relative to ``grid_recovery.py``:

1. **Independent random voxels + post-hoc accuracy reporting** (inspired by
   Kim et al. 2024's "simulated voxels" design, as far as it applies to this
   project's unrelated subcortical model): ``N_VOXELS`` independent,
   continuously (off-grid) sampled ground-truth parameter sets, rather than
   a small ``N_GT_COMBOS`` set reused across every Monte Carlo repeat.
   Accuracy is reported post-hoc (Kim et al.'s Fig. 4E style):
   ``compute_accuracy_correlations`` computes Pearson r between the true
   value and the **per-voxel mean recovered value** (averaged across Monte
   Carlo repeats — see below), across all voxels, one r per parameter — a
   different statistic than the grid search's own fitting/argmax criterion.

2. **Single shared ``tau_ms`` instead of separate ``tau_on_ms``/``tau_off_ms``**:
   a debug-scale smoke test showed these two are not distinguishable from
   EACH OTHER either, not just hard to recover in general — the noiseless
   oracle correlation across every (tau_on, tau_off) combination at the
   debug grid's resolution varied by only ~0.005 in Pearson r (range
   0.989–0.994), with the supposed "best" candidate not even reliably the
   one closest to the true value. Fitting them as two independent free
   parameters doesn't reflect what's actually identifiable post-HRF, so
   this script fits one shared ``tau_ms`` (``tau_on_ms = tau_off_ms =
   tau_ms`` is passed into ``forward_bold_runs``) instead. This is scoped
   to this script only — ``grid_recovery.py``'s original 5-parameter
   (split tau_on/tau_off) results are already written up in its own
   SUMMARY.md and left as-is.

Per project discussion, this version ALSO reverted from "one noisy draw per
independent voxel" back to Monte Carlo noise repeats per voxel (reusing
``grid_recovery.N_NOISE_REPEATS``) — a single draw has no averaging to
stabilize a fit that's already near-degenerate (see point 2 above), so it
produced results that looked like near-arbitrary picks among tied candidates
rather than graceful degradation. ``compute_accuracy_correlations`` uses the
per-voxel MEAN recovered value across repeats as that voxel's point estimate
for the accuracy-correlation table.

Deliberately NOT replicated (per project discussion, unchanged from the
original design): BADS continuous refinement (fitting stays grid-search
Pearson r argmax), the paper's single fixed-SNR noise calibration (keeps
this project's existing low/mid/high ``NOISE_VOXELS`` sweep), and its
design-seed sweep has no analogue in the paper but is kept here (reusing
``grid_recovery.DESIGN_SEEDS``) since this project already checks whether
findings hold across trial orderings.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery

(``--no-capture-output`` matters: plain ``conda run`` buffers/withholds the
child process's stdout until it exits — even with ``python -u`` — so without
it, this script's per-alpha/per-voxel progress prints stay invisible for the
whole candidate-grid build, which can look exactly like a hang. See project
discussion.)
"""

import time
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
    N_NOISE_REPEATS,
    ALPHA_MIN,
    ALPHA_MAX,
    ALPHA_GRID,
    W_MIN,
    W_MAX,
    W_GRID,
    TAU_ON_GRID_MS,
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)

# ── This script's own sweep size — independent random voxels (see module
# docstring). First-pass value per project discussion. ──
N_VOXELS = 100

# Shared ON/OFF time constant -- reuses grid_recovery.py's existing tau_on
# grid verbatim as the single tau grid (same values, same range), rather
# than inventing a new one -- see module docstring for why tau_on/tau_off
# were unified into one parameter.
TAU_GRID_MS = list(TAU_ON_GRID_MS)

PARAM_NAMES = ["cf_index", "tau_ms", "alpha", "w"]

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


def build_candidate_grid(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate every (cf_index, tau_ms, alpha, w) candidate.

    ``tau_ms`` is passed as BOTH ``tau_on_ms`` and ``tau_off_ms`` into
    ``forward_bold_runs`` (shared ON/OFF time constant — see module
    docstring). One fewer free dimension than ``grid_recovery.py``'s 5D
    grid, so this candidate grid is ``len(TAU_ON_GRID_MS)``x smaller for the
    same CF/alpha/w resolution.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4)
        Columns ``[cf_index, tau_ms, alpha, w]``.
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(TAU_GRID_MS) * len(ALPHA_GRID) * len(W_GRID)
    print(f"Building candidate grid: {n_cf} CF x {len(TAU_GRID_MS)} tau x "
          f"{len(ALPHA_GRID)} alpha x {len(W_GRID)} w = {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 4), dtype=float)
    template_matrix = None
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        for cf_index in range(n_cf):
            per_seq, cf_hz = build_per_seq_trains(
                raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
                TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
            )
            for tau_ms in TAU_GRID_MS:
                for w in W_GRID:
                    bold_per_run = forward_bold_runs(
                        per_seq, run_designs, hrf_kernel, cf_hz,
                        TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
                        w, tau_ms, tau_ms, RECTIFY, RHO,
                    )
                    template = np.concatenate(bold_per_run)
                    if template_matrix is None:
                        template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
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


def sample_ground_truths_off_grid(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """Draw ground-truth combos with tau_ms/alpha/w OFF the candidate grid.

    CF_index is still drawn on-grid (inherently discrete). Off-grid only
    (no on-grid mode here) — the natural match for "independently,
    randomly sampled within bounds" voxels.

    Returns
    -------
    np.ndarray, shape (n_combos, 4)
        Columns ``[cf_index, tau_ms, alpha, w]``.
    """
    cf_index = rng.integers(0, n_cf, size=n_combos).astype(float)
    tau_ms   = rng.uniform(TAU_GRID_MS[0], TAU_GRID_MS[-1], size=n_combos)
    alpha    = 2 ** rng.uniform(np.log2(ALPHA_MIN), np.log2(ALPHA_MAX), size=n_combos)
    w        = rng.uniform(W_MIN, W_MAX, size=n_combos)
    return np.column_stack([cf_index, tau_ms, alpha, w])


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
    """Monte Carlo recovery loop for one noise level, given already-sampled
    independent voxels — same MC structure as ``grid_recovery.run_recovery``
    (``N_NOISE_REPEATS`` noisy draws per voxel, batched via one matrix
    multiply), just applied per-voxel over ``N_VOXELS`` independent ground
    truths instead of a small reused ``N_GT_COMBOS`` set, and with the
    unified 4-parameter model.

    Returns
    -------
    recovered_params : np.ndarray, shape (n_voxels, N_NOISE_REPEATS, 4)
    """
    n_voxels = true_params.shape[0]
    recovered_params = np.empty((n_voxels, N_NOISE_REPEATS, 4), dtype=float)

    for v_i, (cf_index_f, tau_ms, alpha, w) in enumerate(true_params):
        cf_index = int(round(cf_index_f))
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        per_seq, cf_hz = build_per_seq_trains(
            raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
            TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
        )
        gt_bold_per_run = forward_bold_runs(
            per_seq, run_designs, hrf_kernel, cf_hz,
            TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
            w, tau_ms, tau_ms, RECTIFY, RHO,
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

        bias = recovered_params[v_i].mean(axis=0) - [cf_index, tau_ms, alpha, w]
        print(f"  [{noise_voxel}] voxel {v_i+1}/{n_voxels}: true cf_index={cf_index} "
              f"tau={tau_ms:.1f} alpha={alpha:.3f} w={w:.3f} | bias={np.round(bias, 3)}")

    return recovered_params


def compute_accuracy_correlations(
    true_params: np.ndarray, recovered_params: np.ndarray, param_names: list = PARAM_NAMES,
) -> dict:
    """Post-hoc accuracy statistic, Kim et al. 2024 Fig. 4E style: Pearson r
    between true values and the PER-VOXEL MEAN recovered value (averaged
    across the ``N_NOISE_REPEATS`` Monte Carlo draws), across all
    independent voxels, one r per parameter.

    This is NOT the grid search's own fitting criterion (that stays Pearson
    r argmax against the candidate grid, computed per repeat inside
    ``run_recovery``) — this is a separate, after-the-fact summary of how
    well recovery tracked ground truth across voxels.

    Parameters
    ----------
    recovered_params : np.ndarray, shape (n_voxels, n_repeats, n_params)
    """
    recovered_mean = recovered_params.mean(axis=1)
    return {
        name: float(np.corrcoef(true_params[:, p], recovered_mean[:, p])[0, 1])
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


def plot_accuracy_comparison(accuracy_by_noise: dict, save_path: Path, param_names: list = PARAM_NAMES,
                              n_voxels: int = N_VOXELS, n_repeats: int = N_NOISE_REPEATS) -> None:
    """Bar chart of the post-hoc accuracy statistic: one group per parameter,
    one bar per noise level, height = Pearson r between true values and
    per-voxel mean recovered values across all independent voxels (Fig. 4E
    style).

    ``n_voxels``/``n_repeats`` are for the title only — pass them explicitly
    rather than relying on this module's own ``N_VOXELS``/``N_NOISE_REPEATS``
    defaults when called from a different script with its own counts (e.g.
    kim2024_style_duration_recovery.py) — otherwise the title would silently
    show THIS module's values instead of the caller's.
    """
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
    ax.set_ylabel("Pearson r (true vs. per-voxel mean recovered)")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.legend(title="noise level")
    ax.set_title(f"Recovery accuracy across noise levels (N={n_voxels} voxels, "
                 f"{n_repeats:,} MC repeats each)")

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

        gt_rng = np.random.default_rng(MC_SEED)
        true_params = sample_ground_truths_off_grid(gt_rng, n_cf, N_VOXELS)

        recovered_by_noise = {}
        accuracy_by_noise = {}
        for noise_voxel in NOISE_VOXELS:
            print(f"\nRunning recovery at design_seed={design_seed}, noise_voxel='{noise_voxel}': "
                  f"{N_VOXELS} independent voxels x {N_NOISE_REPEATS:,} MC noise repeats each ...")
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
