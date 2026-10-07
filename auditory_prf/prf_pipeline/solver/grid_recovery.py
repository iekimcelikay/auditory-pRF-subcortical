"""grid_recovery.py
====================
Joint 5D grid-search parameter-recovery test for the tone-cloud AdapTrans+BOLD
pipeline: CF_index x tau_on_ms x tau_off_ms x alpha x w.

Validation logic (see project discussion — this is the standard model-based
pRF "template matching" procedure, e.g. Dumoulin & Wandell 2008, run on
simulated data with a known ground truth so recovery accuracy is checkable):

  1. Build the full noiseless candidate-template grid once (cached — see
     ``forward_model.py``'s module docstring for why loading/sharpening/
     chunking are cached at coarser granularity than the full 5D grid).
  2. For each GT_SAMPLING_MODES entry ("on_grid": every true parameter sits
     exactly on a candidate-grid point, the simpler first test; "off_grid":
     tau/alpha/w drawn continuously instead, also exposing grid-resolution/
     quantization error): draw N_GT_COMBOS ground-truth parameter
     combinations (CF_index is always on-grid — it is inherently discrete).
  3. For each ground truth, forward-simulate the noiseless "perfect" BOLD,
     then for each NOISE_VOXELS entry ("low"/"mid"/"high"): Monte Carlo loop
     (>= N_NOISE_REPEATS independent PmNoise draws per ground truth) — add
     noise to the ground truth -> "measurement", Pearson-correlate against
     every noiseless candidate template, argmax -> recovered parameter
     combination. The same ground truths and noise seeds are reused across
     noise levels (only the amplitude differs), so the three levels are
     directly comparable.
  4. Recovery scatter plot per parameter per (gt_mode, noise_voxel): true (y)
     vs. recovered (x), pooled across all ground-truth combos and repeats.
     Plus one noise-level-comparison plot per gt_mode, overlaying low/mid/high
     for the same ground truths.

Usage (from project root):
    conda run -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.grid_recovery
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

# ── Paths ──────────────────────────────────────────────────────────────────
# TODO(confirm): point this at the freshly-regenerated dIPC cochlear output
# (stage-1 run); directory name assumed to mirror the stimuli timestamp.
RESULTS_DIR       = Path("models_output/toneclouds_gaussianprf_20261005_1741")

# Stamped once per process run (not per function call) so every invocation
# of this script gets its own timestamped subdirectory (and filenames) under
# the stable OUTPUT_DIR — nothing gets overwritten by a later run, and a
# result file is still traceable to its run even if copied out on its own.
RUN_TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR          = Path("grid_recovery_results")
RUN_DIR             = OUTPUT_DIR / RUN_TIMESTAMP
CANDIDATE_GRID_PATH = RUN_DIR / f"candidate_grid_{RUN_TIMESTAMP}.npz"           # mode/noise-independent, built once


def recovery_output_paths(gt_mode: str, noise_voxel: str) -> tuple[Path, Path, Path]:
    """Per-(GT-sampling-mode, noise-level) output paths — see GT_SAMPLING_MODES, NOISE_VOXELS."""
    tag = f"{gt_mode}_{noise_voxel}_{RUN_TIMESTAMP}"
    return (
        RUN_DIR / f"grid_recovery_results_{tag}.npz",
        RUN_DIR / f"grid_recovery_scatter_{tag}.png",
        RUN_DIR / f"grid_recovery_table_{tag}.csv",
    )


def noise_comparison_plot_path(gt_mode: str) -> Path:
    """Path for the per-GT-mode noise-level comparison figure (see plot_noise_comparison)."""
    return RUN_DIR / f"grid_recovery_noise_comparison_{gt_mode}_{RUN_TIMESTAMP}.png"

# ── Fixed run/timing parameters — reused verbatim from
# full_pipeline_toneclouds_adaptrans_claude.py (the real toneclouds_gaussianprf
# experiment design: 24 active conditions (8 tone_on_ms x 3 Gaussian-filterbank
# centers) + NULL_FRACTION null trials, back-to-back with zero ITI jitter,
# 6.4s opening/closing blanks -> 652.8s/run) rather than tau_recovery_sanity_check.py's
# ad-hoc random-sampled design, which does not match this stimulus family. ──
TR_S            = 1.6
TOTAL_RUN_DUR_S = 652.8    # opening_blank_s (6.4) + 32 trials x 20s (640) + closing_blank_s (6.4)
SIGNAL_DT_S     = 1e-3
RECTIFY         = True
RHO             = 1.0
NOISE_VOXELS    = ["low", "mid", "high"]   # swept — one recovery run per level, same ground truths
CHUNK_MARGIN_MS = 30.0     # matches full_pipeline_toneclouds_adaptrans_claude.py's
                           # convention for this stimulus family (NOT
                           # chunk_from_id's own 50 ms default).

# Number of runs used to build each candidate/ground-truth forward simulation
# for THIS recovery test — intentionally decoupled from the real experiment's
# N_RUNS=24 (which would make the candidate-grid build take ~24x longer; see
# conversation). Start at 1 run to keep the grid build tractable; adjustable —
# planned to increase later to test recovery under different run-design scales.
N_RUNS_RECOVERY = 1

# ── Candidate grid (fitting side — stays noiseless) ─────────────────────────
TAU_ON_GRID_MS  = [20, 30, 50, 75, 100, 150, 250, 300, 400]
TAU_OFF_GRID_MS = [20, 30, 50, 75, 100, 150, 250, 300, 400]
ALPHA_MIN       = 1.0
ALPHA_MAX       = 8.0      # adjustable
ALPHA_K_STEP    = 0.5      # exponent step; adjustable for a finer search later
W_MIN, W_MAX, W_STEP = 0.5, 0.9, 0.05

_k_min, _k_max = np.log2(ALPHA_MIN), np.log2(ALPHA_MAX)
ALPHA_GRID = (2 ** np.arange(_k_min, _k_max + ALPHA_K_STEP / 2, ALPHA_K_STEP)).tolist()
W_GRID     = np.round(np.arange(W_MIN, W_MAX + W_STEP / 2, W_STEP), 4).tolist()

# ── Ground-truth sweep / Monte Carlo sizing (adjustable — see conversation:
# both are explicitly meant to be tuned, not fixed) ─────────────────────────
N_GT_COMBOS     = 20
N_NOISE_REPEATS = 10_000   # per ground-truth combo, per user instruction ("at least 10,000")
MC_SEED         = 2026     # root seed for ground-truth sampling + noise spawning

# "on_grid"  -> sample_ground_truths_on_grid  (first, simpler test: can recovery
#               work at all when the true value sits exactly on a grid point)
# "off_grid" -> sample_ground_truths_off_grid (harder test: tau/alpha/w drawn
#               continuously, also exposing grid-resolution/quantization error)
# Swept — one full (GT sampling x noise level) run per mode, each its own
# (independent) set of sampled ground truths (unlike NOISE_VOXELS, these two
# modes can't share one true_params array — on-grid and off-grid draws are
# different tests, not the same ground truths viewed under different noise).
GT_SAMPLING_MODES = ["on_grid", "off_grid"]

PARAM_NAMES = ["cf_index", "tau_on_ms", "tau_off_ms", "alpha", "w"]


def build_candidate_grid(
    raw_sequences: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    hrf_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate every (cf_index, tau_on, tau_off, alpha, w) candidate.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 5)
        Columns ``[cf_index, tau_on_ms, tau_off_ms, alpha, w]``.
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
        Each row is one candidate's noiseless concatenated-BOLD template,
        zero-meaned and unit-normalized — ready for a single matrix multiply
        against a (batch of) normalized noisy measurement(s) to get Pearson r.
    """
    n_cf = len(cf_list_hz)
    n_candidates = n_cf * len(TAU_ON_GRID_MS) * len(TAU_OFF_GRID_MS) * len(ALPHA_GRID) * len(W_GRID)
    print(f"Building candidate grid: {n_cf} CF x {len(TAU_ON_GRID_MS)} tau_on x "
          f"{len(TAU_OFF_GRID_MS)} tau_off x {len(ALPHA_GRID)} alpha x {len(W_GRID)} w "
          f"= {n_candidates:,} candidates")

    candidate_params = np.empty((n_candidates, 5), dtype=float)
    template_matrix = None  # allocated once n_timepoints is known
    idx = 0
    t_start = time.time()

    for alpha in ALPHA_GRID:
        sharpened_by_seq = sharpen_all_sequences(raw_sequences, alpha)
        for cf_index in range(n_cf):
            per_seq, cf_hz = build_per_seq_trains(
                raw_sequences, sharpened_by_seq, cf_index, cf_list_hz,
                TC_SILENCE_SEQ_ID, chunk_margin_ms=CHUNK_MARGIN_MS,
            )
            for tau_on_ms in TAU_ON_GRID_MS:
                for tau_off_ms in TAU_OFF_GRID_MS:
                    for w in W_GRID:
                        bold_per_run = forward_bold_runs(
                            per_seq, run_designs, hrf_kernel, cf_hz,
                            TOTAL_RUN_DUR_S, TR_S, SIGNAL_DT_S,
                            w, tau_on_ms, tau_off_ms, RECTIFY, RHO,
                        )
                        template = np.concatenate(bold_per_run)
                        if template_matrix is None:
                            template_matrix = np.empty((n_candidates, len(template)), dtype=np.float32)
                        template_matrix[idx, :] = template
                        candidate_params[idx] = (cf_index, tau_on_ms, tau_off_ms, alpha, w)
                        idx += 1
        elapsed = time.time() - t_start
        print(f"  alpha={alpha:.3f} done ({idx:,}/{n_candidates:,} candidates, "
              f"{elapsed:.0f}s elapsed)")

    template_matrix -= template_matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
    template_matrix /= np.where(norms == 0, 1.0, norms)

    return candidate_params, template_matrix


def sample_ground_truths_on_grid(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """Draw ground-truth combos with every parameter sitting exactly on a candidate-grid point.

    First, simpler recovery test: can the procedure recover a parameter at
    all when the true value conveniently coincides with one of the fitting
    grid's own points — before testing the harder off-grid case (see
    ``sample_ground_truths_off_grid``).

    Returns
    -------
    np.ndarray, shape (n_combos, 5)
        Columns ``[cf_index, tau_on_ms, tau_off_ms, alpha, w]``.
    """
    cf_index = rng.integers(0, n_cf, size=n_combos).astype(float)
    tau_on   = rng.choice(TAU_ON_GRID_MS, size=n_combos).astype(float)
    tau_off  = rng.choice(TAU_OFF_GRID_MS, size=n_combos).astype(float)
    alpha    = rng.choice(ALPHA_GRID, size=n_combos)
    w        = rng.choice(W_GRID, size=n_combos)
    return np.column_stack([cf_index, tau_on, tau_off, alpha, w])


def sample_ground_truths_off_grid(rng: np.random.Generator, n_cf: int, n_combos: int) -> np.ndarray:
    """Draw ground-truth combos with tau_on/tau_off/alpha/w OFF the candidate grid.

    CF_index is still drawn on-grid (it is inherently discrete — no finer
    resolution exists between cochlear channels). tau_on/tau_off/alpha/w are
    drawn uniformly within range instead, so the recovery test also reveals
    grid-resolution/quantization error, not just identifiability — the
    harder test to run after ``sample_ground_truths_on_grid`` looks good.

    Returns
    -------
    np.ndarray, shape (n_combos, 5)
        Columns ``[cf_index, tau_on_ms, tau_off_ms, alpha, w]``.
    """
    cf_index = rng.integers(0, n_cf, size=n_combos).astype(float)
    tau_on   = rng.uniform(TAU_ON_GRID_MS[0], TAU_ON_GRID_MS[-1], size=n_combos)
    tau_off  = rng.uniform(TAU_OFF_GRID_MS[0], TAU_OFF_GRID_MS[-1], size=n_combos)
    alpha    = 2 ** rng.uniform(np.log2(ALPHA_MIN), np.log2(ALPHA_MAX), size=n_combos)
    w        = rng.uniform(W_MIN, W_MAX, size=n_combos)
    return np.column_stack([cf_index, tau_on, tau_off, alpha, w])


def sample_ground_truths(rng: np.random.Generator, n_cf: int, n_combos: int, gt_mode: str) -> np.ndarray:
    """Draw ground-truth combos once, via whichever mode ``gt_mode`` selects.

    Called once per (GT-sampling-mode) — not once per noise level — the
    whole point of sweeping NOISE_VOXELS is comparing recovery for the
    *same* ground truths across noise levels, so sampling must happen
    outside the per-noise-level recovery loop (see ``run_recovery``).
    """
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

    Parameters
    ----------
    true_params : np.ndarray, shape (n_gt_combos, 5)
        Ground truths to recover — sampled once via ``sample_ground_truths``
        and reused identically across every noise level, so recovery quality
        at 'low' vs. 'mid' vs. 'high' is comparable (same combos, same noise
        seeds, only the noise amplitude differs).
    noise_voxel : str
        ``PmNoise`` preset for this run — ``'low'``, ``'mid'``, or ``'high'``.

    Returns
    -------
    recovered_params : np.ndarray, shape (n_gt_combos, n_noise_repeats, 5)
    """
    n_gt_combos = true_params.shape[0]
    recovered_params = np.empty((n_gt_combos, N_NOISE_REPEATS, 5), dtype=float)

    for gt_i, (cf_index_f, tau_on_ms, tau_off_ms, alpha, w) in enumerate(true_params):
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

        # Batch all repeats' noisy measurements into one matrix, then one
        # matrix-matrix multiply against the candidate grid (BLAS GEMM) —
        # far faster than N_NOISE_REPEATS individual matrix-vector products.
        # Same noise seeds every noise level (keyed on gt_i/rep, not
        # noise_voxel) — only the PmNoise amplitude preset differs, so any
        # difference in recovered estimates across levels is attributable to
        # noise amplitude, not to drawing different random noise.
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

        r = template_matrix_norm @ measurements.T   # (n_candidates, n_noise_repeats)
        best_idx = np.argmax(r, axis=0)              # (n_noise_repeats,)
        recovered_params[gt_i, :, :] = candidate_params[best_idx]

        bias = recovered_params[gt_i].mean(axis=0) - [cf_index, tau_on_ms, tau_off_ms, alpha, w]
        print(f"  [{noise_voxel}] GT {gt_i+1}/{n_gt_combos}: cf_index={cf_index} tau_on={tau_on_ms:.1f} "
              f"tau_off={tau_off_ms:.1f} alpha={alpha:.3f} w={w:.3f} | bias={np.round(bias, 3)}")

    return recovered_params


def plot_recovery(true_params: np.ndarray, recovered_params: np.ndarray, save_path: Path) -> None:
    """Per-parameter recovery scatter: true (y) vs. recovered (x), unity line.

    Points are colored by ground-truth combo index, consistently across all
    panels — so a point's color identifies which combo (and therefore which
    values of the *other* 4 parameters) it came from. Look those up in the
    companion table (see ``save_recovery_table``) rather than guessing from
    the plot alone.
    """
    n_gt_combos, n_repeats, n_params = recovered_params.shape
    # Continuous colormap sampled evenly across n_gt_combos, rather than a
    # fixed-size categorical one (e.g. tab20): tab20's 20 "distinct" colors
    # are hue-paired (light/dark variants of the same base hue) and become
    # hard to tell apart once several combos' points overlap — a continuous
    # hue sweep stays maximally spread out for any n_gt_combos, with no hard
    # ceiling where colors start repeating or converging.
    cmap = plt.get_cmap("hsv")
    colors = [cmap(gt_i / max(n_gt_combos, 1)) for gt_i in range(n_gt_combos)]

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    for p, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        for gt_i in range(n_gt_combos):
            est_vals = recovered_params[gt_i, :, p]
            true_val = np.full(n_repeats, true_params[gt_i, p])
            ax.scatter(est_vals, true_val, s=6, alpha=0.3, color=colors[gt_i],
                       label=f"GT {gt_i}" if p == 0 else None)

        true_vals_all = np.repeat(true_params[:, p], n_repeats)
        est_vals_all  = recovered_params[:, :, p].ravel()
        lo = min(true_vals_all.min(), est_vals_all.min())
        hi = max(true_vals_all.max(), est_vals_all.max())
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, zorder=0)
        ax.set_xlabel(f"recovered {name}")
        ax.set_ylabel(f"true {name}")
        ax.set_title(name)

    handles, labels = axes[0].get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)   # legend swatches stay readable regardless of the data points' alpha
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=7, title="GT combo",
               ncol=max(1, int(np.ceil(n_gt_combos / 20))))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Recovery scatter saved -> {save_path.resolve()}")


def save_recovery_table(true_params: np.ndarray, recovered_params: np.ndarray, save_path: Path) -> None:
    """Save a per-ground-truth-combo CSV: true value, recovered mean, and bias for every parameter.

    Companion to ``plot_recovery`` — the GT combo index (``gt_i``) matches
    that plot's legend/color, so a point's exact context (all 5 true
    parameter values at once) can be looked up by combo index rather than
    read off the scatter.
    """
    n_gt_combos = true_params.shape[0]
    header = ["gt_combo"]
    for name in PARAM_NAMES:
        header += [f"true_{name}", f"recovered_mean_{name}", f"bias_{name}"]

    rows = []
    for gt_i in range(n_gt_combos):
        row = [gt_i]
        for p in range(len(PARAM_NAMES)):
            true_val = true_params[gt_i, p]
            recovered_mean = recovered_params[gt_i, :, p].mean()
            row += [true_val, recovered_mean, recovered_mean - true_val]
        rows.append(row)

    with open(save_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
    print(f"Recovery table saved -> {save_path.resolve()}")


def plot_noise_comparison(
    true_params: np.ndarray,
    recovered_by_noise: dict,
    save_path: Path,
) -> None:
    """Per-parameter recovery scatter, overlaying all noise levels for the SAME ground truths.

    Unlike ``plot_recovery`` (which colors by ground-truth combo, for one
    noise level), here color encodes noise level and every ground-truth
    combo's points are pooled together — the point of this figure is seeing
    how recovery degrades from 'low' to 'high' noise, not tracing individual
    combos (use ``plot_recovery``/the per-level CSV table for that).

    Parameters
    ----------
    true_params : np.ndarray, shape (n_gt_combos, 5)
    recovered_by_noise : dict[str, np.ndarray]
        ``{noise_voxel: recovered_params}``, each shape (n_gt_combos, n_noise_repeats, 5).
    """
    n_params = true_params.shape[1]
    noise_levels = list(recovered_by_noise.keys())
    level_colors = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    for p, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        all_vals = [true_params[:, p]]
        for noise_voxel in noise_levels:
            recovered_params = recovered_by_noise[noise_voxel]
            n_repeats = recovered_params.shape[1]
            true_vals = np.repeat(true_params[:, p], n_repeats)
            est_vals  = recovered_params[:, :, p].ravel()
            all_vals.append(est_vals)
            ax.scatter(est_vals, true_vals, s=5, alpha=0.15,
                       color=level_colors.get(noise_voxel, "steelblue"),
                       label=noise_voxel if p == 0 else None)

        lo = min(v.min() for v in all_vals)
        hi = max(v.max() for v in all_vals)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, zorder=0)
        ax.set_xlabel(f"recovered {name}")
        ax.set_ylabel(f"true {name}")
        ax.set_title(name)

    handles, labels = axes[0].get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)   # legend swatches stay readable regardless of the data points' alpha
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title="noise level")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Noise-level comparison plot saved -> {save_path.resolve()}")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw cochlear PSTH sequences from {RESULTS_DIR} ...")
    raw_sequences, cf_list_hz = load_raw_sequences(RESULTS_DIR)
    n_cf = len(cf_list_hz)
    print(f"  {len(raw_sequences)} sequences loaded | {n_cf} CFs")

    # Real toneclouds_gaussianprf trial structure (see
    # full_pipeline_toneclouds_adaptrans_claude.py:run_pipeline): 24 active
    # conditions (8 tone_on_ms x 3 Gaussian-filterbank centers) + null trials.
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

    # Random-seeded, not the real experiment's fixed BASE_SEED=42 — offset
    # into the same "distinct from BASE_SEED" pool used by the existing
    # gt35_singlerun_worker.py / gt_onoff_singlerun_worker.py design-seed
    # sweeps (DESIGN_SEEDS = [1000 + i for i in range(N_DESIGN_SEEDS)]),
    # so this can never collide with the real pipeline's per-run seeds.
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

    # One independent GT sample per mode (on-grid vs. off-grid are different
    # tests, not the same ground truths under different noise — see
    # GT_SAMPLING_MODES), but reused identically across every noise level
    # within that mode — see run_recovery's docstring for why that part
    # can't happen inside the noise-level loop.
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
