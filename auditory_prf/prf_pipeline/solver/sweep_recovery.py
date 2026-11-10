"""sweep_recovery.py
====================
Kim-2024-style parameter recovery for the FM-sweep forward model
(power-law sharpening -> AdapTrans over the whole run -> HRF; run_assembly_sweeps.py).

Recovered parameters: cf_index, alpha, tau_ms, w (rho fixed at 1.0, tau_on == tau_off).

Same recipe as the tone-cloud recovery scripts:
  1. Noiseless candidate grid (30 CF x 7 alpha x 9 tau x 4 w = 7,560 templates of one
     408-TR run); templates mean-removed and unit-normalised.
  2. Ground-truth voxels: ``on_grid`` (random candidate rows; a miss then comes from
     identifiability, not quantisation) or ``off_grid`` (continuous draws in the grid's range).
  3. Measurements: the clean GT BOLD plus PmNoise (``--noise low|mid|high``, N repeats)
     or the clean BOLD itself (``--noise none``, one repeat).
  4. Fit: Pearson r against every template, argmax -> recovered parameters (the "old"
     argmax structure: recovered_params of shape (n_voxels, n_repeats, n_params)).
  5. Accuracy: Pearson r between true and per-voxel mean recovered values across voxels.

Added for the sweeps, computed on the CLEAN GT BOLD of every voxel (independent of noise):
  - r_best: correlation of the best candidate (1.0 for noise-free on-grid)
  - margin_<param>: r_best minus the best r among candidates whose <param> differs from the
    winner's, i.e. how distinguishable that parameter is (small margin = needs low noise)
  - n_ties: candidates within TIE_TOL of r_best

Several noise levels share one grid and the same ground truths (like the tone-cloud
``noisy_ongrid`` runners); the comparison plots and accuracy table cover all levels.

Run from the project root:
    python -m auditory_prf.prf_pipeline.solver.sweep_recovery --noise none --gt-mode on_grid
    python -m auditory_prf.prf_pipeline.solver.sweep_recovery --noise all --gt-mode on_grid \\
        --grid-file sweep_recovery_results/<run>/candidate_grid_design1000.npz
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.prf_pipeline.hrf import SUBCORTICAL_PARAMS, build_hrf_kernel
from auditory_prf.prf_pipeline.powerlaw_function import compute_population_scale
from auditory_prf.prf_pipeline.run_assembly_sweeps import (
    SIGNAL_DT_S,
    assemble_sweep_run_train,
    load_raw_psths,
    run_sweep_forward_model,
    sharpen_cf_rows,
    sweep_run_bold,
)
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import (
    ALPHA_GRID,
    ALPHA_MAX,
    ALPHA_MIN,
    MC_SEED,
    N_NOISE_REPEATS,
    TAU_ON_GRID_MS,
    plot_noise_comparison,
    plot_recovery,
    save_recovery_table,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    plot_accuracy_comparison,
    save_accuracy_table,
)
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels

logger = logging.getLogger(__name__)

PARAM_NAMES = ["cf_index", "alpha", "tau_ms", "w"]
TAU_GRID_MS = list(TAU_ON_GRID_MS)            # 20-400 ms, tau_on == tau_off
W_GRID = [0.7, 0.75, 0.8, 0.85]               # same as the combined tone-cloud recovery
W_MIN, W_MAX = 0.7, 0.85
RECTIFY = True
RHO = 1.0
N_VOXELS = 100
DESIGN_SEED = 1000
N_TOP_CANDIDATES = 5
TIE_TOL = 1e-5                                 # templates are float32
NOISE_JITTER = [0.0, 0.0, 0.0]                 # PmNoise jitter (freq, amp, phase), as the noisy_ongrid runners
NOISE_LEVELS = ["low", "mid", "high"]


# ── Candidate grid ────────────────────────────────────────────────────────────
_grid_state: dict = {}


def _templates_for_alpha_cf(task: tuple) -> tuple:
    """Worker: all (tau, w) templates of one (alpha, cf_index); returns (params, templates)."""
    alpha, cf_index = task
    s = _grid_state
    per_seq_train = sharpen_cf_rows(s["raw_psth_by_seq"], cf_index, alpha, s["scales"][alpha])
    spont_rate = float(per_seq_train[s["silence_seq_id"]].mean())
    full_train = assemble_sweep_run_train(per_seq_train, s["run_design"], s["total_run_dur_s"],
                                          spont_rate)
    params, templates = [], []
    for tau_ms in s["tau_grid_ms"]:
        for w in s["w_grid"]:
            bold = sweep_run_bold(full_train, s["hrf_kernel"], s["tr_s"], w=w, rectify=RECTIFY,
                                  rho=RHO, tau_ms=tau_ms)["bold_combined"]
            params.append((cf_index, alpha, tau_ms, w))
            templates.append(bold)
    return np.array(params, dtype=float), np.array(templates, dtype=np.float32)


def build_candidate_grid(
    raw_psth_by_seq: dict,
    n_cf: int,
    run_design: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    alpha_grid: list = ALPHA_GRID,
    tau_grid_ms: list = TAU_GRID_MS,
    w_grid: list = W_GRID,
    n_workers: int = 1,
    normalize: bool = True,
) -> tuple:
    """Forward-simulate every (cf_index, alpha, tau_ms, w) candidate (noiseless).

    Each (alpha, cf_index) run train is assembled once; (tau, w) only change AdapTrans
    and the HRF stage.

    Parameters
    ----------
    normalize : bool
        True: unit-norm templates (one run, matrix product = Pearson r). False: only
        mean-removed, so each run keeps its amplitude and several runs can be concatenated
        and normalised once over the whole concatenation.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4), order alpha > CF > tau > w
    template_matrix_norm : np.ndarray float32, shape (n_candidates, n_timepoints)
        Mean-removed templates, unit-norm if ``normalize``.
    """
    n_candidates = n_cf * len(alpha_grid) * len(tau_grid_ms) * len(w_grid)
    logger.info("Building candidate grid: %d CF x %d alpha x %d tau x %d w = %d candidates",
                n_cf, len(alpha_grid), len(tau_grid_ms), len(w_grid), n_candidates)
    scales = {alpha: compute_population_scale(raw_psth_by_seq.values(), alpha)
              for alpha in alpha_grid}
    _grid_state.update(raw_psth_by_seq=raw_psth_by_seq, scales=scales, run_design=run_design,
                       total_run_dur_s=total_run_dur_s, tr_s=tr_s, silence_seq_id=silence_seq_id,
                       hrf_kernel=hrf_kernel, tau_grid_ms=tau_grid_ms, w_grid=w_grid)
    tasks = [(alpha, cf_index) for alpha in alpha_grid for cf_index in range(n_cf)]

    t_start = time.time()
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(processes=n_workers) as pool:   # forked: state inherited
            results = list(pool.imap(_templates_for_alpha_cf, tasks, chunksize=1))
    else:
        results = [_templates_for_alpha_cf(task) for task in tasks]
    candidate_params = np.concatenate([params for params, _ in results])
    template_matrix = np.concatenate([templates for _, templates in results])
    assert candidate_params.shape == (n_candidates, len(PARAM_NAMES))
    logger.info("Grid built in %.0f s; template matrix %s", time.time() - t_start,
                template_matrix.shape)

    template_matrix -= template_matrix.mean(axis=1, keepdims=True)
    if normalize:
        norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
        template_matrix /= np.where(norms == 0, 1.0, norms)
    return candidate_params, template_matrix


# ── Ground truths ─────────────────────────────────────────────────────────────
def sample_ground_truths(
    gt_mode: str,
    rng: np.random.Generator,
    n_cf: int,
    candidate_params: np.ndarray,
    n_voxels: int,
) -> np.ndarray:
    """Ground-truth (cf_index, alpha, tau_ms, w) rows.

    on_grid: random candidate-grid rows. off_grid: cf_index uniform over the CFs, alpha and
    tau_ms log-uniform over the grid's range (independent of CF), w uniform.

    Returns
    -------
    np.ndarray, shape (n_voxels, 4)
    """
    assert gt_mode in ("on_grid", "off_grid"), gt_mode
    if gt_mode == "on_grid":
        return candidate_params[rng.integers(0, len(candidate_params), size=n_voxels)].copy()
    cf_index = rng.integers(0, n_cf, size=n_voxels).astype(float)
    alpha = 2 ** rng.uniform(np.log2(ALPHA_MIN), np.log2(ALPHA_MAX), size=n_voxels)
    tau_ms = 2 ** rng.uniform(np.log2(min(TAU_GRID_MS)), np.log2(max(TAU_GRID_MS)), size=n_voxels)
    w = rng.uniform(W_MIN, W_MAX, size=n_voxels)
    return np.column_stack([cf_index, alpha, tau_ms, w])


# ── Recovery ──────────────────────────────────────────────────────────────────
def clean_margins(gt_bold: np.ndarray, template_matrix_norm: np.ndarray,
                  candidate_params: np.ndarray, tie_tol: float = TIE_TOL) -> dict:
    """Identifiability of one voxel from its CLEAN BOLD against the whole grid.

    Returns
    -------
    dict with best_idx, r_best, margins (n_params,), n_ties, top_idx / top_r (N_TOP_CANDIDATES,)
    """
    y = gt_bold.astype(np.float64)
    y -= y.mean()
    y /= np.linalg.norm(y)
    r = template_matrix_norm.astype(np.float64) @ y
    best_idx = int(np.argmax(r))
    margins = np.full(candidate_params.shape[1], np.nan)
    for p in range(candidate_params.shape[1]):
        differs = ~np.isclose(candidate_params[:, p], candidate_params[best_idx, p])
        if differs.any():
            margins[p] = r[best_idx] - r[differs].max()
    top_idx = np.argsort(r)[::-1][:N_TOP_CANDIDATES]
    return {"best_idx": best_idx, "r_best": float(r[best_idx]), "margins": margins,
            "n_ties": int((r >= r[best_idx] - tie_tol).sum()),
            "top_idx": top_idx, "top_r": r[top_idx]}


def _recover_one_voxel(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker task: GT BOLD, all noise repeats, argmax fit, clean-BOLD margins."""
    cf_index, alpha, tau_ms, w = true_row
    gt_bold = run_sweep_forward_model(
        state["raw_psth_by_seq"], int(round(cf_index)), state["run_design"],
        state["total_run_dur_s"], state["tr_s"], state["silence_seq_id"],
        alpha=alpha, w=w, rectify=RECTIFY, rho=RHO, tau_ms=tau_ms,
        hrf_kernel=state["hrf_kernel"])["bold_combined"]

    n_repeats = state["n_repeats"]
    measurements = np.empty((n_repeats, len(gt_bold)), dtype=np.float32)
    for rep in range(n_repeats):
        run_seeds = spawn_noise_seeds(MC_SEED, v_i, rep, n_runs=1)
        measurements[rep, :] = draw_noisy_measurement([gt_bold], state["noise"], state["tr_s"],
                                                      run_seeds, jitter=state["jitter"])
    best_idx = best_match_indices(state["template_matrix_norm"], measurements)
    result = clean_margins(gt_bold, state["template_matrix_norm"], state["candidate_params"])
    result["recovered"] = state["candidate_params"][best_idx]          # (n_repeats, n_params)
    return result


def run_recovery(
    raw_psth_by_seq: dict,
    run_design: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    n_workers: int = 1,
    jitter: Optional[list] = NOISE_JITTER,
) -> dict:
    """Monte Carlo recovery for one noise level.

    Returns
    -------
    dict with recovered_params (n_voxels, n_repeats, n_params) and per-voxel arrays
    best_idx, r_best, n_ties (n_voxels,), margins (n_voxels, n_params),
    top_idx / top_r (n_voxels, N_TOP_CANDIDATES).
    """
    state = dict(raw_psth_by_seq=raw_psth_by_seq, run_design=run_design,
                 total_run_dur_s=total_run_dur_s, tr_s=tr_s, silence_seq_id=silence_seq_id,
                 hrf_kernel=hrf_kernel, candidate_params=candidate_params,
                 template_matrix_norm=template_matrix_norm, noise=noise, n_repeats=n_repeats,
                 jitter=jitter)
    if n_workers > 1:
        voxel_results = list(map_voxels(_recover_one_voxel, true_params, state, n_workers))
    else:
        voxel_results = [_recover_one_voxel(v_i, row, state) for v_i, row in enumerate(true_params)]
    return {
        "recovered_params": np.stack([r["recovered"] for r in voxel_results]),
        "best_idx": np.array([r["best_idx"] for r in voxel_results]),
        "r_best": np.array([r["r_best"] for r in voxel_results]),
        "n_ties": np.array([r["n_ties"] for r in voxel_results]),
        "margins": np.stack([r["margins"] for r in voxel_results]),
        "top_idx": np.stack([r["top_idx"] for r in voxel_results]),
        "top_r": np.stack([r["top_r"] for r in voxel_results]),
    }


def run_noise_sweep(
    raw_psth_by_seq: dict,
    run_design: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    true_params: np.ndarray,
    noise_levels: list,
    n_repeats_by_noise: dict,
    n_workers: int = 1,
    jitter: Optional[list] = NOISE_JITTER,
) -> dict:
    """run_recovery at every noise level on the SAME ground truths; {noise: result}."""
    results = {}
    for noise in noise_levels:
        logger.info("noise '%s': %d voxels x %d repeats ...", noise, len(true_params),
                    n_repeats_by_noise[noise])
        results[noise] = run_recovery(
            raw_psth_by_seq, run_design, total_run_dur_s, tr_s, silence_seq_id, hrf_kernel,
            candidate_params, template_matrix_norm, true_params, noise,
            n_repeats_by_noise[noise], n_workers, jitter)
    return results


LOG_SPACED_PARAMS = ("alpha", "tau_ms")      # grid steps are measured in log space for these


def _level_index(values: np.ndarray, levels: np.ndarray, log_spaced: bool) -> np.ndarray:
    """Index of the nearest grid level for every value (log distance if log_spaced)."""
    v, l = (np.log(values), np.log(levels)) if log_spaced else (values, levels)
    return np.abs(v[..., None] - l).argmin(axis=-1)


def summarize(true_params: np.ndarray, result: dict, candidate_params: np.ndarray,
              cf_list_hz: Optional[np.ndarray] = None) -> dict:
    """Printable summary of one noise level.

    Per parameter: exact_fraction (recovered value == true value), within_1_step (recovered
    grid level at most one level from the true one), median_step_error (in grid levels),
    margin_median / margin_min, and relative_mse:
      relative_mse = mean over voxels v and repeats k of ((rec[v, k] - true[v]) / true[v])^2
    i.e. every error is divided by its own voxel's true value before squaring and averaging.
    cf_index has no meaningful nominal (index 0 is a real CF), so when cf_list_hz is given the
    CF in Hz is used instead of the index; without it relative_mse is NaN if a true value is 0.
    relative_mse_by_voxel: (voxels,) array per parameter, the mean over repeats for each voxel
    (relative_mse is the mean of these).
    chance_relative_mse: the same statistic for a guess drawn uniformly from the grid levels,
    independent of the true value (same true voxels); recovery is only informative where it
    beats this.
    Also exact_joint: all parameters exact in the same repeat.
    candidate_params defines the grid levels (unique values per parameter).
    """
    recovered = result["recovered_params"]
    keys = ("exact_fraction", "within_1_step", "median_step_error", "margin_median", "margin_min",
            "relative_mse", "chance_relative_mse", "relative_mse_by_voxel")
    summary = {key: {} for key in keys}
    exact = np.isclose(recovered, true_params[:, None, :])               # (voxels, repeats, params)
    for p, name in enumerate(PARAM_NAMES):
        levels = np.unique(candidate_params[:, p])
        log = name in LOG_SPACED_PARAMS
        step_error = np.abs(_level_index(recovered[:, :, p], levels, log)
                            - _level_index(true_params[:, p], levels, log)[:, None])
        summary["exact_fraction"][name] = float(exact[:, :, p].mean())
        summary["within_1_step"][name] = float((step_error <= 1).mean())
        summary["median_step_error"][name] = float(np.median(step_error))
        summary["margin_median"][name] = float(np.nanmedian(result["margins"][:, p]))
        summary["margin_min"][name] = float(np.nanmin(result["margins"][:, p]))
        true_value, recovered_value, guess_levels = true_params[:, p], recovered[:, :, p], levels
        if name == "cf_index" and cf_list_hz is not None:
            index_axis = np.arange(len(cf_list_hz))
            true_value = np.interp(true_value, index_axis, cf_list_hz)
            recovered_value = np.interp(recovered_value, index_axis, cf_list_hz)
            guess_levels = np.interp(levels, index_axis, cf_list_hz)
        error = recovered_value - true_value[:, None]
        # chance: a guess drawn uniformly from the grid levels, independent of the true value;
        # the expectation over the guess is the mean over levels (exact, no sampling)
        guess_error = guess_levels[None, :] - true_value[:, None]            # (voxels, levels)
        nominal_defined = bool(np.all(true_value != 0))                      # division by true needs it
        relative_mse_by_voxel = (                                            # (voxels,): mean over repeats
            ((error / true_value[:, None]) ** 2).mean(axis=1) if nominal_defined
            else np.full(len(true_value), np.nan))
        summary["relative_mse_by_voxel"][name] = relative_mse_by_voxel
        summary["relative_mse"][name] = float(relative_mse_by_voxel.mean())   # same K per voxel
        summary["chance_relative_mse"][name] = (
            float(((guess_error / true_value[:, None]) ** 2).mean()) if nominal_defined else np.nan)
    summary["exact_joint"] = float(exact.all(axis=2).mean())
    summary["n_ties_max"] = int(result["n_ties"].max())
    return summary


def save_summary_table(summary: dict, save_path: Path) -> None:
    """CSV, one row per parameter plus the joint exact rate."""
    columns = ["exact_fraction", "within_1_step", "median_step_error", "margin_median", "margin_min",
               "relative_mse", "chance_relative_mse"]
    with open(save_path, "w") as f:
        f.write("parameter," + ",".join(columns) + "\n")
        for name in PARAM_NAMES:
            f.write(name + "," + ",".join(f"{summary[c][name]:.6g}" for c in columns) + "\n")
        f.write(f"all_parameters_exact,{summary['exact_joint']:.6g}" + "," * (len(columns) - 1) + "\n")
    logger.info("Summary table saved -> %s", save_path.resolve())


def plot_relative_mse(summary_by_noise: dict, save_path: Path, param_names: list = PARAM_NAMES,
                      n_voxels: Optional[int] = None, n_repeats: Optional[int] = None,
                      log_y: bool = True) -> None:
    """Bar chart of the relative MSE: one group per parameter, one bar per noise level.

    Height = summary["relative_mse"] = mean(((recovered - true) / true)^2); 0 is perfect, the
    dashed line at 1 marks an error as large as the parameter value itself. The black ticks are
    the chance level of each parameter (a guess drawn uniformly from its grid); bars at or
    above a tick are no better than guessing. The values span orders of magnitude, hence log_y.
    """
    metric = "relative_mse"
    level_colors = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
    noise_levels = list(summary_by_noise.keys())
    x = np.arange(len(param_names))
    width = 0.8 / max(len(noise_levels), 1)

    fig, ax = plt.subplots(figsize=(1.5 * len(param_names) + 2, 4))
    for i, noise in enumerate(noise_levels):
        heights = [summary_by_noise[noise][metric][name] for name in param_names]
        bars = ax.bar(x + i * width, heights, width, label=noise,
                      color=level_colors.get(noise, "steelblue"))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0)
    chance_key = f"chance_{metric}"
    first_summary = next(iter(summary_by_noise.values()))       # chance does not depend on noise
    chance = [first_summary[chance_key][name] for name in param_names]
    for j, value in enumerate(chance):
        ax.hlines(value, x[j] - 0.5 * width, x[j] + (len(noise_levels) - 0.5) * width,
                  color="black", linewidth=2.2, zorder=4,
                  label="chance (uniform grid guess)" if j == 0 else None)
    ax.set_xticks(x + width * (len(noise_levels) - 1) / 2)
    ax.set_xticklabels(param_names)
    ax.set_ylabel("relative MSE  [mean((recovered - true)$^2$ / true$^2$)]")
    all_values = ([s[metric][n] for s in summary_by_noise.values() for n in param_names]
                  + [v for v in chance if np.isfinite(v)])
    if log_y:
        ax.set_yscale("log")
        positive = [v for v in all_values if v > 0]              # log axis cannot show 0
        ax.set_ylim(min(positive, default=1e-3) / 2, max(1.0, max(all_values)) * 3)
    else:
        ax.set_ylim(0, max(1.1, 1.1 * max(all_values)))
    ax.legend(title="noise level", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    size = f" (N={n_voxels} voxels, {n_repeats:,} MC repeats each)" if n_voxels and n_repeats else ""
    ax.set_title(f"Relative MSE across noise levels{size}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Relative MSE plot saved -> %s", save_path.resolve())


RELATIVE_MSE_PLOT_FLOOR = 1e-4      # voxels with exactly 0 error are drawn here on the log axis


def plot_relative_mse_distribution(summary_by_noise: dict, save_path: Path,
                                   param_names: list = PARAM_NAMES,
                                   n_voxels: Optional[int] = None,
                                   n_repeats: Optional[int] = None) -> None:
    """Distribution of the per-voxel relative MSE: one panel per parameter, one box per noise level.

    Every dot is one voxel (summary["relative_mse_by_voxel"]: its mean squared relative error over
    the repeats); the box shows the quartiles, the white diamond the mean over voxels (= the bar
    height of plot_relative_mse). Black line: chance level; dashed line: relative MSE of 1.
    The y-axis is shared and logarithmic, so panels are comparable with the chance level.
    """
    level_colors = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
    noise_levels = list(summary_by_noise.keys())
    first_summary = next(iter(summary_by_noise.values()))       # chance does not depend on noise
    rng = np.random.default_rng(0)                              # fixed jitter, reproducible figure

    fig, axes = plt.subplots(1, len(param_names), sharey=True,
                             figsize=(2.6 * len(param_names) + 2, 4.2))
    for ax, name in zip(np.atleast_1d(axes), param_names):
        for i, noise in enumerate(noise_levels):
            values = np.maximum(summary_by_noise[noise]["relative_mse_by_voxel"][name],
                                RELATIVE_MSE_PLOT_FLOOR)
            color = level_colors.get(noise, "steelblue")
            ax.boxplot(values, positions=[i], widths=0.55, showfliers=False, manage_ticks=False,
                       boxprops=dict(color=color), whiskerprops=dict(color=color),
                       capprops=dict(color=color), medianprops=dict(color=color, linewidth=2))
            ax.scatter(i + rng.uniform(-0.18, 0.18, size=len(values)), values, s=9, alpha=0.55,
                       color=color, linewidths=0, zorder=3)
            ax.scatter(i, values.mean(), marker="D", s=40, facecolor="white", edgecolor="black",
                       zorder=5, label="mean over voxels" if (i == 0 and name == param_names[0]) else None)
        ax.axhline(first_summary["chance_relative_mse"][name], color="black", linewidth=2.2,
                   zorder=4, label="chance (uniform grid guess)" if name == param_names[0] else None)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels(noise_levels)
        ax.set_xlim(-0.6, len(noise_levels) - 0.4)
        ax.set_title(name)
    axes[0].set_ylabel("per-voxel relative MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    n_floored = sum(int((v < RELATIVE_MSE_PLOT_FLOOR).sum()) for s in summary_by_noise.values()
                    for v in s["relative_mse_by_voxel"].values())
    if n_floored:
        labels.append(f"values < {RELATIVE_MSE_PLOT_FLOOR:g} drawn at {RELATIVE_MSE_PLOT_FLOOR:g}")
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color="gray", markersize=3))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    size = f" (N={n_voxels} voxels, {n_repeats:,} MC repeats each)" if n_voxels and n_repeats else ""
    fig.suptitle(f"Per-voxel relative MSE = mean over repeats of ((recovered - true) / true)$^2$"
                 f"{size}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Relative MSE distribution plot saved -> %s", save_path.resolve())


def plot_relative_mse_by_true_value(summary_by_noise: dict, true_params: np.ndarray,
                                    candidate_params: np.ndarray, save_path: Path,
                                    param_name: str = "tau_ms", n_repeats: Optional[int] = None) -> None:
    """Per-voxel relative MSE grouped by the TRUE value of one parameter (default tau_ms).

    One group per grid level of param_name, one box per noise level; every dot is a voxel whose
    true value is that level (summary["relative_mse_by_voxel"]). Black tick: chance level at that
    true value, mean over the grid levels g of ((g - true) / true)^2. Dashed line: relative MSE 1.
    Not for cf_index (its relative MSE is computed on the CF in Hz, not on the index).
    """
    assert param_name != "cf_index", "cf_index is evaluated in Hz; group by its Hz value instead"
    p = PARAM_NAMES.index(param_name)
    level_colors = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
    noise_levels = list(summary_by_noise.keys())
    levels = np.unique(candidate_params[:, p])
    true_value = true_params[:, p]
    n_noise = len(noise_levels)
    offsets = (np.arange(n_noise) - (n_noise - 1) / 2) * 0.27
    rng = np.random.default_rng(0)                              # fixed jitter, reproducible figure

    fig, ax = plt.subplots(figsize=(1.3 * len(levels) + 3, 4.6))
    for g, level in enumerate(levels):
        in_group = np.isclose(true_value, level)
        for i, noise in enumerate(noise_levels):
            values = np.maximum(summary_by_noise[noise]["relative_mse_by_voxel"][param_name][in_group],
                                RELATIVE_MSE_PLOT_FLOOR)
            if len(values) == 0:
                continue
            color = level_colors.get(noise, "steelblue")
            ax.boxplot(values, positions=[g + offsets[i]], widths=0.22, showfliers=False,
                       manage_ticks=False, boxprops=dict(color=color),
                       whiskerprops=dict(color=color), capprops=dict(color=color),
                       medianprops=dict(color=color, linewidth=2))
            ax.scatter(g + offsets[i] + rng.uniform(-0.07, 0.07, size=len(values)), values, s=9,
                       alpha=0.6, color=color, linewidths=0, zorder=3,
                       label=noise if g == 0 else None)
        chance = float((((levels - level) / level) ** 2).mean())
        ax.hlines(chance, g - 0.45, g + 0.45, color="black", linewidth=2.2, zorder=4,
                  label="chance (uniform grid guess)" if g == 0 else None)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{level:g}\n(n={int(np.isclose(true_value, level).sum())})"
                        for level in levels])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_xlabel(f"true {param_name} (n = voxels with that true value)")
    ax.set_ylabel("per-voxel relative MSE")
    ax.legend(title="noise level", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    size = f" ({n_repeats:,} MC repeats per voxel)" if n_repeats else ""
    ax.set_title(f"Per-voxel relative MSE by true {param_name}{size}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Relative MSE by true %s plot saved -> %s", param_name, save_path.resolve())


def save_margin_table(true_params: np.ndarray, result: dict, save_path: Path) -> None:
    """Per-voxel CSV: true params, winner params (first repeat), r_best, n_ties, margins."""
    recovered = result["recovered_params"][:, 0, :]
    header = (["gt_combo"] + [f"true_{n}" for n in PARAM_NAMES]
              + [f"winner_{n}" for n in PARAM_NAMES] + ["r_best", "n_ties"]
              + [f"margin_{n}" for n in PARAM_NAMES])
    with open(save_path, "w") as f:
        f.write(",".join(header) + "\n")
        for v in range(len(true_params)):
            row = ([v] + list(true_params[v]) + list(recovered[v])
                   + [result["r_best"][v], result["n_ties"][v]] + list(result["margins"][v]))
            f.write(",".join(f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x)
                             for x in row) + "\n")
    logger.info("Margin table saved -> %s", save_path.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=str,
                        default="models_output/sweep_design_erb_3band/combined")
    parser.add_argument("--output-dir", type=str, default="sweep_recovery_results")
    parser.add_argument("--noise", nargs="+", default=["none"],
                        choices=["none", "low", "mid", "high", "all"],
                        help="One or more noise levels; 'all' = low mid high, run on the same "
                             "ground truths with one shared grid.")
    parser.add_argument("--gt-mode", choices=["on_grid", "off_grid"], default="on_grid")
    parser.add_argument("--n-voxels", type=int, default=N_VOXELS)
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Per noise level. Default: 1 for 'none', else N_NOISE_REPEATS.")
    parser.add_argument("--design-seed", type=int, default=DESIGN_SEED)
    parser.add_argument("--n-workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--grid-file", type=str, default=None,
                        help="Load a saved candidate_grid_*.npz instead of rebuilding.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import stimuli.sweep_design_erb as design   # needs the project root on sys.path
    noise_levels = NOISE_LEVELS if "all" in args.noise else list(dict.fromkeys(args.noise))
    n_repeats_by_noise = {noise: args.n_repeats or (1 if noise == "none" else N_NOISE_REPEATS)
                          for noise in noise_levels}
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = (Path(args.output_dir) / f"{timestamp}_{'-'.join(noise_levels)}_"
               f"{args.gt_mode}_design{args.design_seed}")   # seed keeps array tasks apart
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("noise=%s gt_mode=%s voxels=%d repeats=%s workers=%d -> %s", noise_levels,
                args.gt_mode, args.n_voxels, n_repeats_by_noise, args.n_workers, run_dir)

    raw_psth_by_seq, cf_list_hz = load_raw_psths(Path(args.results_dir))
    run_design = design.build_run_design(args.design_seed)
    total_run_dur_s = design.total_run_duration_s(run_design)
    tr_s = design.TR_S
    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)
    n_cf = len(cf_list_hz)

    if args.grid_file:
        with np.load(args.grid_file) as grid:
            candidate_params, template_matrix_norm = grid["candidate_params"], grid["template_matrix_norm"]
    else:
        candidate_params, template_matrix_norm = build_candidate_grid(
            raw_psth_by_seq, n_cf, run_design, total_run_dur_s, tr_s, design.SILENCE_SEQ_ID,
            hrf_kernel, n_workers=args.n_workers)
        np.savez(run_dir / f"candidate_grid_design{args.design_seed}.npz",
                 candidate_params=candidate_params, template_matrix_norm=template_matrix_norm)

    rng = np.random.default_rng(MC_SEED)
    true_params = sample_ground_truths(args.gt_mode, rng, n_cf, candidate_params, args.n_voxels)
    results = run_noise_sweep(raw_psth_by_seq, run_design, total_run_dur_s, tr_s,
                              design.SILENCE_SEQ_ID, hrf_kernel, candidate_params,
                              template_matrix_norm, true_params, noise_levels,
                              n_repeats_by_noise, args.n_workers)

    recovered_by_noise, accuracy_by_noise, summary_by_noise = {}, {}, {}
    for noise, result in results.items():
        recovered = result["recovered_params"]
        recovered_by_noise[noise] = recovered
        accuracy = compute_accuracy_correlations(true_params, recovered, param_names=PARAM_NAMES)
        accuracy_by_noise[noise] = accuracy
        summary = summarize(true_params, result, candidate_params, cf_list_hz)
        summary_by_noise[noise] = summary
        logger.info("[%s] accuracy (true vs per-voxel mean recovered, Pearson r): %s", noise,
                    {k: round(v, 4) for k, v in accuracy.items()})
        logger.info("[%s] exact recovery fraction: %s", noise,
                    {k: round(v, 3) for k, v in summary["exact_fraction"].items()})
        logger.info("[%s] recovered within 1 grid step: %s", noise,
                    {k: round(v, 3) for k, v in summary["within_1_step"].items()})
        logger.info("[%s] median grid-step error: %s | all four exact in the same repeat: %.4f",
                    noise, summary["median_step_error"], summary["exact_joint"])
        logger.info("[%s] relative MSE (mean((rec - true)^2 / true^2); cf in Hz): %s", noise,
                    {k: round(v, 4) for k, v in summary["relative_mse"].items()})
        logger.info("[%s] chance-level relative MSE (uniform grid guess): %s", noise,
                    {k: round(v, 4) for k, v in summary["chance_relative_mse"].items()})
        logger.info("[%s] clean-BOLD margin median: %s", noise,
                    {k: f"{v:.2e}" for k, v in summary["margin_median"].items()})
        logger.info("[%s] clean-BOLD margin min   : %s", noise,
                    {k: f"{v:.2e}" for k, v in summary["margin_min"].items()})
        logger.info("[%s] max candidates tied with the best (r within %.0e): %d", noise,
                    TIE_TOL, summary["n_ties_max"])

        # the old argmax structure, per noise level, then the margins
        # name follows replot_from_results.py: <prefix>_results_design<N>_<level>_<timestamp>.npz
        np.savez(run_dir / f"sweep_recovery_results_design{args.design_seed}_{noise}_{timestamp}.npz",
                 true_params=true_params, recovered_params=recovered,
                 param_names=PARAM_NAMES,
                 **{k: v for k, v in result.items() if k != "recovered_params"})
        plot_recovery(true_params, recovered, run_dir / f"recovery_scatter_{noise}.png",
                      param_names=PARAM_NAMES)
        save_recovery_table(true_params, recovered, run_dir / f"recovery_table_{noise}.csv",
                            param_names=PARAM_NAMES)
        save_margin_table(true_params, result, run_dir / f"margin_table_{noise}.csv")
        save_summary_table(summary, run_dir / f"summary_table_{noise}.csv")

    save_accuracy_table(accuracy_by_noise, run_dir / f"accuracy_table_design{args.design_seed}.csv",
                        param_names=PARAM_NAMES)
    plot_accuracy_comparison(accuracy_by_noise,
                             run_dir / f"accuracy_comparison_design{args.design_seed}.png",
                             param_names=PARAM_NAMES, n_voxels=args.n_voxels,
                             n_repeats=max(n_repeats_by_noise.values()))
    plot_relative_mse(summary_by_noise, run_dir / f"relative_mse_design{args.design_seed}.png",
                      param_names=PARAM_NAMES, n_voxels=args.n_voxels,
                      n_repeats=max(n_repeats_by_noise.values()))
    plot_relative_mse_distribution(
        summary_by_noise, run_dir / f"relative_mse_distribution_design{args.design_seed}.png",
        param_names=PARAM_NAMES, n_voxels=args.n_voxels, n_repeats=max(n_repeats_by_noise.values()))
    plot_relative_mse_by_true_value(
        summary_by_noise, true_params, candidate_params,
        run_dir / f"relative_mse_by_tau_design{args.design_seed}.png",
        param_name="tau_ms", n_repeats=max(n_repeats_by_noise.values()))
    if len(noise_levels) > 1:
        plot_noise_comparison(true_params, recovered_by_noise,
                              run_dir / f"noise_comparison_design{args.design_seed}.png",
                              param_names=PARAM_NAMES)
    logger.info("done -> %s", run_dir.resolve())


if __name__ == "__main__":
    main()
