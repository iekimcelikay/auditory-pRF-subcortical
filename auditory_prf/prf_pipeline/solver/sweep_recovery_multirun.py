"""sweep_recovery_multirun.py
===========================
Recovery of (cf_index, alpha, tau_ms, w) from SEVERAL runs with different stimulus orders
(the real experiment has 24), using all runs jointly (method B1: concatenation).

For one candidate the template is the concatenation of its clean BOLD over the run orders
(each run mean-removed, amplitude kept), normalised once over the whole concatenation; the
measurement is the noisy runs concatenated (independent noise per run, each run mean-removed);
the winner is the candidate with the highest Pearson r over the whole concatenation.

One noise-free candidate grid per run order (``build-grid``, one SLURM array task each),
then ``recover`` loads them all. Design: auditory-prf-docs-all/sweep_24run_recovery_design.md.

Run from the project root.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.prf_pipeline.hrf import SUBCORTICAL_PARAMS, build_hrf_kernel
from auditory_prf.prf_pipeline.run_assembly_sweeps import (
    SIGNAL_DT_S,
    load_raw_psths,
    run_sweep_forward_model,
)
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import (
    MC_SEED,
    plot_recovery,
    save_recovery_table,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    plot_accuracy_comparison,
    save_accuracy_table,
)
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    N_VOXELS,
    NOISE_JITTER,
    PARAM_NAMES,
    RECTIFY,
    RHO,
    build_candidate_grid,
    clean_margins,
    plot_relative_mse,
    plot_relative_mse_by_true_value,
    plot_relative_mse_distribution,
    sample_ground_truths,
    save_margin_table,
    save_summary_table,
    summarize,
)

logger = logging.getLogger(__name__)

RUN_DESIGN_SEEDS = tuple(range(1000, 1024))      # 24 run orders, one per design seed
RUN_PREFIX = "sweep_recovery_24runs"             # result-file prefix (see replot_from_results.py)
N_REPEATS = 1000                                 # Monte Carlo repeats per noise level (one repeat = one experiment)
DEFAULT_RESULTS_DIR = "models_output/sweep_design_erb_3band/combined"
DEFAULT_GRID_DIR = "sweep_recovery_multirun_grids"
DEFAULT_OUTPUT_DIR = "sweep_recovery_results"
ESTIMATES = ("joint", "random_effects")          # the two 24-run estimates


# ── Run designs and per-run grids ─────────────────────────────────────────────
def build_run_designs(design_seeds: Sequence[int]) -> tuple:
    """Run orders of the given design seeds.

    Returns
    -------
    run_designs : list of run designs (each a list of (soundfileid, onset_s))
    total_run_dur_s : float
        Common run length in seconds (asserted equal for every run).
    """
    import stimuli.sweep_design_erb as design   # needs the project root on sys.path
    run_designs = [design.build_run_design(seed) for seed in design_seeds]
    durations_s = [design.total_run_duration_s(run_design) for run_design in run_designs]
    assert np.allclose(durations_s, durations_s[0]), f"run durations differ: {durations_s}"
    return run_designs, float(durations_s[0])


def save_run_grid(save_path: Path, candidate_params: np.ndarray,
                  template_matrix_mean_removed: np.ndarray) -> None:
    """Save one run order's grid: candidate_params (n_cand, 4), templates (n_cand, n_tr)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, candidate_params=candidate_params,
             template_matrix_mean_removed=template_matrix_mean_removed)
    logger.info("Run grid saved -> %s", save_path.resolve())


def load_run_grids(grid_dir: Path, design_seeds: Sequence[int]) -> tuple:
    """Load ``run_grid_design<seed>.npz`` for every seed.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4)
    template_per_run : list of np.ndarray float32, each (n_candidates, n_tr), mean-removed
    """
    candidate_params, template_per_run = None, []
    for seed in design_seeds:
        path = Path(grid_dir) / f"run_grid_design{seed}.npz"
        with np.load(path) as grid:
            params = grid["candidate_params"]
            templates = grid["template_matrix_mean_removed"].astype(np.float32)
        if candidate_params is None:
            candidate_params = params
        assert np.array_equal(params, candidate_params), (
            f"candidate order of {path.name} differs from the first grid "
            f"(design{design_seeds[0]})")
        assert templates.shape == (len(candidate_params), templates.shape[1]), path.name
        assert not template_per_run or templates.shape == template_per_run[0].shape, (
            f"{path.name}: {templates.shape[1]} TRs, expected {template_per_run[0].shape[1]}")
        template_per_run.append(templates)
    return candidate_params, template_per_run


def concatenate_templates(template_per_run: list) -> np.ndarray:
    """Concatenate the runs' templates and normalise once over the whole concatenation.

    Parameters
    ----------
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)
        Clean BOLD per run; mean is removed per run here (amplitude is kept).

    Returns
    -------
    np.ndarray float32, shape (n_candidates, n_runs * n_tr)
        Rows have mean 0 and unit norm over the concatenation (an all-zero row stays zero).
    """
    assert all(t.shape == template_per_run[0].shape for t in template_per_run), (
        "runs differ in shape: " + str([t.shape for t in template_per_run]))
    template_matrix = np.concatenate(
        [t - t.mean(axis=1, keepdims=True) for t in template_per_run], axis=1).astype(np.float32)
    norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
    return template_matrix / np.where(norms == 0, 1.0, norms)


# ── Per-run templates, random-effects average, spread ─────────────────────────
def normalize_run_templates(template_per_run: list) -> list:
    """Unit-norm templates of every run on its own (the single-run recovery's templates).

    Parameters
    ----------
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)

    Returns
    -------
    list of np.ndarray float32, each (n_candidates, n_tr), rows mean 0 and unit norm
    (an all-zero row stays zero).
    """
    normalized = []
    for templates in template_per_run:
        centered = templates - templates.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        normalized.append((centered / np.where(norms == 0, 1.0, norms)).astype(np.float32))
    return normalized


def _to_average_space(estimates: np.ndarray, cf_list_hz: np.ndarray) -> np.ndarray:
    """(..., 4) estimates -> natural log of CF in Hz, alpha and tau_ms; w unchanged."""
    assert estimates.shape[-1] == 4, estimates.shape
    cf_index = np.round(estimates[..., 0]).astype(int)
    assert cf_index.min() >= 0 and cf_index.max() < len(cf_list_hz), "cf_index outside the CF list"
    return np.stack([np.log(cf_list_hz[cf_index]), np.log(estimates[..., 1]),
                     np.log(estimates[..., 2]), estimates[..., 3]], axis=-1)


def average_run_estimates(recovered_per_run: np.ndarray, cf_list_hz: np.ndarray) -> np.ndarray:
    """Random effects: average the per-run estimates (log space for CF in Hz, alpha, tau; w linear).

    Parameters
    ----------
    recovered_per_run : np.ndarray, shape (..., n_runs, 4)
        (cf_index, alpha, tau_ms, w) recovered from each run alone.
    cf_list_hz : np.ndarray, shape (n_cf,), increasing

    Returns
    -------
    np.ndarray, shape (..., 4)
        The average; cf_index is generally fractional (the log-mean CF mapped back onto the
        index by interpolation), alpha and tau_ms are geometric means, w the arithmetic mean.
    """
    assert np.all(np.diff(cf_list_hz) > 0), "cf_list_hz must be increasing"
    mean_space = _to_average_space(recovered_per_run, cf_list_hz).mean(axis=-2)
    cf_index = np.interp(mean_space[..., 0], np.log(cf_list_hz), np.arange(len(cf_list_hz)))
    return np.stack([cf_index, np.exp(mean_space[..., 1]), np.exp(mean_space[..., 2]),
                     mean_space[..., 3]], axis=-1)


def estimate_spread(recovered_per_run: np.ndarray, cf_list_hz: np.ndarray) -> np.ndarray:
    """Width of the per-run estimates: standard deviation (ddof 0) across runs in the space of
    the averaging (natural log of CF in Hz, alpha and tau_ms; w linear).

    Parameters
    ----------
    recovered_per_run : np.ndarray, shape (..., n_runs, 4)

    Returns
    -------
    np.ndarray, shape (..., 4)
    """
    return _to_average_space(recovered_per_run, cf_list_hz).std(axis=-2)


def chance_spread(candidate_params: np.ndarray, cf_list_hz: np.ndarray) -> np.ndarray:
    """Spread of a uniform guess over the grid levels of each parameter (same space as
    ``estimate_spread``): the standard deviation (ddof 0) of the unique levels.

    Returns
    -------
    np.ndarray, shape (4,)
    """
    cf_levels_hz = cf_list_hz[np.unique(candidate_params[:, 0]).astype(int)]
    return np.array([np.std(np.log(cf_levels_hz)),
                     np.std(np.log(np.unique(candidate_params[:, 1]))),
                     np.std(np.log(np.unique(candidate_params[:, 2]))),
                     np.std(np.unique(candidate_params[:, 3]))])


# ── 24-run recovery (joint fit and random effects) ────────────────────────────
def draw_multirun_measurement(
    clean_bold_per_run: list,
    noise: str,
    tr_s: float,
    voxel_index: int,
    repeat_index: int,
    jitter: Optional[list] = NOISE_JITTER,
) -> np.ndarray:
    """One noisy experiment: independent noise per run, each run mean-removed, concatenated.

    Run r of repeat k of voxel v gets the seed ``spawn_noise_seeds(MC_SEED, v, k, n_runs)[r]``;
    the seed of run 0 does not depend on ``n_runs`` (so run 0 equals the single-run noise).

    Parameters
    ----------
    clean_bold_per_run : list of np.ndarray, each (n_tr,)
    noise : str
        'low' | 'mid' | 'high', or 'none' for the clean BOLD.

    Returns
    -------
    np.ndarray, shape (n_runs * n_tr,)
    """
    n_runs = len(clean_bold_per_run)
    assert len({len(bold) for bold in clean_bold_per_run}) == 1, "runs differ in length"
    run_seeds = spawn_noise_seeds(MC_SEED, voxel_index, repeat_index, n_runs=n_runs)
    noisy_per_run = draw_noisy_measurement(clean_bold_per_run, noise, tr_s, run_seeds,
                                           jitter=jitter).reshape(n_runs, -1)
    return (noisy_per_run - noisy_per_run.mean(axis=1, keepdims=True)).ravel()


def _recover_one_voxel_multirun(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: clean BOLD of every run, all noise repeats, joint and per-run argmax, margins."""
    cf_index, alpha, tau_ms, w = true_row
    clean_bold_per_run = [
        run_sweep_forward_model(
            state["raw_psth_by_seq"], int(round(cf_index)), run_design, state["total_run_dur_s"],
            state["tr_s"], state["silence_seq_id"], alpha=alpha, w=w, rectify=RECTIFY, rho=RHO,
            tau_ms=tau_ms, hrf_kernel=state["hrf_kernel"])["bold_combined"]
        for run_design in state["run_designs"]]
    n_runs, n_tr, n_repeats = len(clean_bold_per_run), len(clean_bold_per_run[0]), state["n_repeats"]
    template_per_run_norm, candidate_params = state["template_per_run_norm"], state["candidate_params"]
    assert n_runs == len(template_per_run_norm)

    measurements = np.empty((n_repeats, n_runs * n_tr), dtype=np.float32)
    for rep in range(n_repeats):
        measurements[rep, :] = draw_multirun_measurement(
            clean_bold_per_run, state["noise"], state["tr_s"], v_i, rep, state["jitter"])

    # per run on its own first: best_match_indices normalises its input in place, so pass copies
    per_run_idx = np.empty((n_repeats, n_runs), dtype=np.int64)
    for r in range(n_runs):
        per_run_idx[:, r] = best_match_indices(
            template_per_run_norm[r], measurements[:, r * n_tr:(r + 1) * n_tr].copy())
    best_idx = best_match_indices(state["template_matrix_norm"], measurements)     # joint

    clean_concatenated = np.concatenate([bold - bold.mean() for bold in clean_bold_per_run])
    result = clean_margins(clean_concatenated, state["template_matrix_norm"], candidate_params)
    result["recovered"] = candidate_params[best_idx]                               # (n_repeats, 4)
    result["recovered_per_run"] = candidate_params[per_run_idx]                    # (n_repeats, n_runs, 4)
    return result


def run_recovery_multirun(
    raw_psth_by_seq: dict,
    run_designs: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    template_per_run_norm: list,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    cf_list_hz: np.ndarray,
    n_workers: int = 1,
    jitter: Optional[list] = NOISE_JITTER,
) -> dict:
    """Monte Carlo recovery from all runs, one noise level: joint fit and random effects.

    Parameters
    ----------
    template_matrix_norm : np.ndarray, shape (n_candidates, n_runs * n_tr)
        ``concatenate_templates`` output (joint fit).
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        ``normalize_run_templates`` output (one fit per run).

    Returns
    -------
    dict with the keys of ``sweep_recovery.run_recovery`` for the JOINT fit
    (recovered_params (n_voxels, n_repeats, 4), best_idx, r_best, n_ties, margins, top_idx, top_r),
    plus recovered_per_run (n_voxels, n_repeats, n_runs, 4), recovered_random_effects
    (n_voxels, n_repeats, 4; the log-space average, see ``average_run_estimates``) and
    recovered_first_run (n_voxels, n_repeats, 4; run 1 alone = the single-run estimate).
    """
    assert len(run_designs) == len(template_per_run_norm), "one template matrix per run order"
    assert template_matrix_norm.shape[1] == len(run_designs) * template_per_run_norm[0].shape[1]
    state = dict(raw_psth_by_seq=raw_psth_by_seq, run_designs=run_designs,
                 total_run_dur_s=total_run_dur_s, tr_s=tr_s, silence_seq_id=silence_seq_id,
                 hrf_kernel=hrf_kernel, candidate_params=candidate_params,
                 template_matrix_norm=template_matrix_norm,
                 template_per_run_norm=template_per_run_norm, noise=noise, n_repeats=n_repeats,
                 jitter=jitter)
    if n_workers > 1:
        voxel_results = list(map_voxels(_recover_one_voxel_multirun, true_params, state, n_workers))
    else:
        voxel_results = [_recover_one_voxel_multirun(v_i, row, state)
                         for v_i, row in enumerate(true_params)]
    recovered_per_run = np.stack([r["recovered_per_run"] for r in voxel_results])
    return {
        "recovered_params": np.stack([r["recovered"] for r in voxel_results]),
        "best_idx": np.array([r["best_idx"] for r in voxel_results]),
        "r_best": np.array([r["r_best"] for r in voxel_results]),
        "n_ties": np.array([r["n_ties"] for r in voxel_results]),
        "margins": np.stack([r["margins"] for r in voxel_results]),
        "top_idx": np.stack([r["top_idx"] for r in voxel_results]),
        "top_r": np.stack([r["top_r"] for r in voxel_results]),
        "recovered_per_run": recovered_per_run,
        "recovered_random_effects": average_run_estimates(recovered_per_run, cf_list_hz),
        "recovered_first_run": recovered_per_run[:, :, 0, :],
    }


# ── Baseline, comparison and spread outputs ───────────────────────────────────
METHOD_LABELS = {"1run": "1 run", "24run_joint": "24 runs, joint fit",
                 "24run_random_effects": "24 runs, random effects"}
METHOD_COLORS = {"1run": "gray", "24run_joint": "steelblue", "24run_random_effects": "mediumpurple"}
NOISE_COLORS = {"low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
SPREAD_AXIS_LABELS = {"cf_index": "cf_index (ln Hz)", "alpha": "alpha (ln)", "tau_ms": "tau_ms (ln)",
                      "w": "w"}


def load_baseline_results(baseline_dir: Path, noise: str, n_repeats: int,
                          true_params: np.ndarray) -> dict:
    """First ``n_repeats`` repeats of the single-run results of one noise level.

    Parameters
    ----------
    baseline_dir : Path
        Folder with ``sweep_recovery_results_design1000_<noise>_<timestamp>.npz``.
    true_params : np.ndarray, shape (n_voxels, 4)
        Ground truths of the new run; asserted equal to the file's (same voxels, same order).

    Returns
    -------
    dict with recovered_params (n_voxels, n_repeats, 4), margins (n_voxels, 4), n_ties (n_voxels,)
    """
    paths = sorted(Path(baseline_dir).glob(f"sweep_recovery_results_design1000_{noise}_*.npz"))
    assert len(paths) == 1, f"expected one baseline file for '{noise}' in {baseline_dir}, found {paths}"
    with np.load(paths[0], allow_pickle=True) as baseline:
        assert baseline["recovered_params"].shape[1] >= n_repeats, (
            f"{paths[0].name} has {baseline['recovered_params'].shape[1]} repeats, need {n_repeats}")
        assert baseline["true_params"].shape == true_params.shape and np.allclose(
            baseline["true_params"], true_params), f"{paths[0].name}: other ground-truth voxels"
        return {"recovered_params": baseline["recovered_params"][:, :n_repeats, :],
                "margins": baseline["margins"], "n_ties": baseline["n_ties"]}


def save_run_comparison_table(summary_by_method: dict, save_path: Path) -> None:
    """Long CSV: one row per noise level x parameter x method.

    summary_by_method : {"1run": {noise: summary}, "24run_joint": ..., "24run_random_effects": ...}
    with summary from ``sweep_recovery.summarize``.
    """
    methods = list(summary_by_method)
    noise_levels = list(summary_by_method[methods[0]])
    columns = ["relative_mse", "chance_relative_mse", "within_1_step", "exact_fraction"]
    with open(save_path, "w") as f:
        f.write("noise,parameter,method," + ",".join(columns) + "\n")
        for noise in noise_levels:
            for name in PARAM_NAMES:
                for method in methods:
                    summary = summary_by_method[method][noise]
                    f.write(f"{noise},{name},{method}," +
                            ",".join(f"{summary[c][name]:.6g}" for c in columns) + "\n")
    logger.info("Run comparison table saved -> %s", Path(save_path).resolve())


def plot_run_comparison(summary_by_method: dict, save_path: Path, n_voxels: Optional[int] = None,
                        n_repeats: Optional[int] = None) -> None:
    """Relative MSE of the methods side by side: one panel per parameter, one group per noise level,
    one bar per method (1 run, joint fit, random effects); black tick = chance level; log y axis.
    """
    methods = list(summary_by_method)
    noise_levels = list(summary_by_method[methods[0]])
    first_summary = summary_by_method[methods[0]][noise_levels[0]]      # chance does not depend on noise
    x = np.arange(len(noise_levels))
    width = 0.8 / len(methods)
    values = [summary_by_method[m][n]["relative_mse"][name] for m in methods for n in noise_levels
              for name in PARAM_NAMES] + [first_summary["chance_relative_mse"][name] for name in PARAM_NAMES]

    fig, axes = plt.subplots(1, len(PARAM_NAMES), sharey=True, figsize=(3.2 * len(PARAM_NAMES) + 2, 4.2))
    for ax, name in zip(axes, PARAM_NAMES):
        for m, method in enumerate(methods):
            heights = [summary_by_method[method][noise]["relative_mse"][name] for noise in noise_levels]
            bars = ax.bar(x + m * width, heights, width, color=METHOD_COLORS.get(method, "steelblue"),
                          label=METHOD_LABELS.get(method, method) if ax is axes[0] else None)
            ax.bar_label(bars, fmt="%.2g", fontsize=6, padding=2)
        ax.hlines(first_summary["chance_relative_mse"][name], x[0] - 0.5 * width,
                  x[-1] + (len(methods) - 0.5) * width, color="black", linewidth=2.2, zorder=4,
                  label="chance (uniform grid guess)" if ax is axes[0] else None)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(noise_levels)
        ax.set_xlabel("noise level")
        ax.set_title(name)
    positive = [v for v in values if v > 0]                      # log axis cannot show 0
    axes[0].set_ylim(min(positive, default=1e-3) / 2, max(1.0, max(values)) * 3)
    axes[0].set_ylabel("relative MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    size = f" (N={n_voxels} voxels, {n_repeats:,} MC repeats each)" if n_voxels and n_repeats else ""
    fig.suptitle(f"Relative MSE: 1 run vs 24 runs{size}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Run comparison plot saved -> %s", Path(save_path).resolve())


def save_spread_table(spread_by_noise: dict, chance_spread_by_param: np.ndarray,
                      save_path: Path) -> None:
    """CSV of the spread of the 24 per-run estimates: median and quartiles over voxels and repeats.

    spread_by_noise : {noise: (n_voxels, n_repeats, 4)} from ``estimate_spread``.
    """
    with open(save_path, "w") as f:
        f.write("noise,parameter,spread_median,spread_q25,spread_q75,chance_spread\n")
        for noise, spread in spread_by_noise.items():
            for p, name in enumerate(PARAM_NAMES):
                q25, median, q75 = np.percentile(spread[..., p], [25, 50, 75])
                f.write(f"{noise},{name},{median:.6g},{q25:.6g},{q75:.6g},"
                        f"{chance_spread_by_param[p]:.6g}\n")
    logger.info("Spread table saved -> %s", Path(save_path).resolve())


def plot_estimate_spread(spread_by_noise: dict, chance_spread_by_param: np.ndarray, save_path: Path,
                         n_voxels: Optional[int] = None, n_repeats: Optional[int] = None) -> None:
    """How wide are the 24 per-run estimates: one panel per parameter, one box per noise level.

    Every dot is a voxel (the median over its repeats of the standard deviation across the 24
    runs, in the space of the averaging); black line = spread of a uniform guess over the grid.
    """
    noise_levels = list(spread_by_noise)
    rng = np.random.default_rng(0)                              # fixed jitter, reproducible figure
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(2.8 * len(PARAM_NAMES) + 2, 4.2))
    for p, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        for i, noise in enumerate(noise_levels):
            per_voxel = np.median(spread_by_noise[noise][..., p], axis=1)        # (n_voxels,)
            color = NOISE_COLORS.get(noise, "steelblue")
            ax.boxplot(per_voxel, positions=[i], widths=0.55, showfliers=False, manage_ticks=False,
                       boxprops=dict(color=color), whiskerprops=dict(color=color),
                       capprops=dict(color=color), medianprops=dict(color=color, linewidth=2))
            ax.scatter(i + rng.uniform(-0.18, 0.18, size=len(per_voxel)), per_voxel, s=9, alpha=0.55,
                       color=color, linewidths=0, zorder=3)
        ax.axhline(chance_spread_by_param[p], color="black", linewidth=2.2, zorder=4,
                   label="chance (uniform grid guess)" if p == 0 else None)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels(noise_levels)
        ax.set_xlim(-0.6, len(noise_levels) - 0.4)
        top = max(chance_spread_by_param[p],
                  max(np.median(s[..., p], axis=1).max() for s in spread_by_noise.values()))
        ax.set_ylim(0, 1.12 * top)                                  # headroom above the chance line
        ax.set_title(SPREAD_AXIS_LABELS[name])
    axes[0].set_ylabel("spread of the 24 per-run estimates (SD)")
    axes[0].legend(loc="lower right", fontsize=7)
    size = f" (N={n_voxels} voxels, {n_repeats:,} MC repeats each)" if n_voxels and n_repeats else ""
    fig.suptitle(f"Width of the 24 per-run estimates{size}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Estimate spread plot saved -> %s", Path(save_path).resolve())


# ── Report and command line ───────────────────────────────────────────────────
def recover_and_report(
    raw_psth_by_seq: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_per_run: list,
    baseline_dir: Path,
    output_dir: Path,
    noise_levels: list,
    n_voxels: int,
    n_repeats: int,
    design_seeds: Sequence[int],
    n_workers: int = 1,
) -> Path:
    """Run the 24-run recovery at every noise level and write all tables, plots and result files.

    Ground truths: the same on-grid draw as ``sweep_recovery`` (``default_rng(MC_SEED)``), so the
    voxels equal the single-run results used as baseline.

    Returns
    -------
    Path of the result folder ``<output_dir>/<timestamp>_<noises>_on_grid_<n>runs``.
    """
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = (Path(output_dir) /
               f"{timestamp}_{'-'.join(noise_levels)}_on_grid_{len(design_seeds)}runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    first_seed = design_seeds[0]
    logger.info("noise=%s voxels=%d repeats=%d runs=%d workers=%d -> %s", noise_levels, n_voxels,
                n_repeats, len(design_seeds), n_workers, run_dir)

    template_matrix_norm = concatenate_templates(template_per_run)
    template_per_run_norm = normalize_run_templates(template_per_run)
    true_params = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(cf_list_hz),
                                       candidate_params, n_voxels)

    summary_by_method = {"1run": {}, "24run_joint": {}, "24run_random_effects": {}}
    estimate_summary = {estimate: {} for estimate in ESTIMATES}
    accuracy = {estimate: {} for estimate in ESTIMATES}
    spread_by_noise = {}
    for noise in noise_levels:
        logger.info("noise '%s': %d voxels x %d repeats x %d runs ...", noise, n_voxels, n_repeats,
                    len(design_seeds))
        result = run_recovery_multirun(
            raw_psth_by_seq, run_designs, total_run_dur_s, tr_s, silence_seq_id, hrf_kernel,
            candidate_params, template_matrix_norm, template_per_run_norm, true_params, noise,
            n_repeats, cf_list_hz, n_workers)
        baseline = load_baseline_results(baseline_dir, noise, n_repeats, true_params)

        identical = float(np.mean(np.all(
            result["recovered_first_run"] == baseline["recovered_params"], axis=-1)))
        logger.info("baseline check [%s]: %.3f of the run-1-alone estimates are identical to the "
                    "single-run results", noise, identical)
        baseline_params = baseline["recovered_params"]
        if identical < 1.0:
            logger.warning("[%s] run 1 alone differs from the single-run results; using the run-1 "
                           "estimates of this run as the 1-run baseline", noise)
            baseline_params = result["recovered_first_run"]

        baseline_result = {"recovered_params": baseline_params, "margins": baseline["margins"],
                           "n_ties": baseline["n_ties"]}
        recovered = {"joint": result["recovered_params"],
                     "random_effects": result["recovered_random_effects"]}
        summaries = {"1run": summarize(true_params, baseline_result, candidate_params, cf_list_hz)}
        for estimate in ESTIMATES:     # the random-effects result reuses the joint margins and ties
            summaries[f"24run_{estimate}"] = summarize(
                true_params, {**result, "recovered_params": recovered[estimate]}, candidate_params,
                cf_list_hz)
        for method, summary in summaries.items():
            summary_by_method[method][noise] = summary
            logger.info("[%s] %s: relative MSE %s | chance %s", noise, method,
                        {k: round(v, 4) for k, v in summary["relative_mse"].items()},
                        {k: round(v, 4) for k, v in summary["chance_relative_mse"].items()})
        spread_by_noise[noise] = estimate_spread(result["recovered_per_run"], cf_list_hz)

        for estimate in ESTIMATES:
            estimate_summary[estimate][noise] = summaries[f"24run_{estimate}"]
            accuracy[estimate][noise] = compute_accuracy_correlations(
                true_params, recovered[estimate], param_names=PARAM_NAMES)
            # file names follow replot_from_results.py: <prefix>_results_design<N>_<level>_<timestamp>.npz
            saved = {k: v for k, v in result.items() if k != "recovered_params"} if estimate == "joint" \
                else {"margins": result["margins"], "n_ties": result["n_ties"]}
            np.savez(run_dir / f"{RUN_PREFIX}_{estimate}_results_design{first_seed}_{noise}_{timestamp}.npz",
                     true_params=true_params, recovered_params=recovered[estimate],
                     param_names=PARAM_NAMES, **saved)
            plot_recovery(true_params, recovered[estimate],
                          run_dir / f"recovery_scatter_{estimate}_{noise}.png", param_names=PARAM_NAMES)
            save_recovery_table(true_params, recovered[estimate],
                                run_dir / f"recovery_table_{estimate}_{noise}.csv", param_names=PARAM_NAMES)
            save_summary_table(summaries[f"24run_{estimate}"],
                               run_dir / f"summary_table_{estimate}_{noise}.csv")
        save_margin_table(true_params, result, run_dir / f"margin_table_{noise}.csv")

    for estimate in ESTIMATES:
        save_accuracy_table(accuracy[estimate], run_dir / f"accuracy_table_{estimate}.csv",
                            param_names=PARAM_NAMES)
        plot_accuracy_comparison(accuracy[estimate], run_dir / f"accuracy_comparison_{estimate}.png",
                                 param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats)
        plot_relative_mse(estimate_summary[estimate], run_dir / f"relative_mse_{estimate}.png",
                          param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats)
        plot_relative_mse_distribution(
            estimate_summary[estimate], run_dir / f"relative_mse_distribution_{estimate}.png",
            param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats)
        plot_relative_mse_by_true_value(
            estimate_summary[estimate], true_params, candidate_params,
            run_dir / f"relative_mse_by_tau_{estimate}.png", param_name="tau_ms", n_repeats=n_repeats)
    save_run_comparison_table(summary_by_method, run_dir / "run_comparison_table.csv")
    plot_run_comparison(summary_by_method, run_dir / "run_comparison.png", n_voxels, n_repeats)
    save_spread_table(spread_by_noise, chance_spread(candidate_params, cf_list_hz),
                      run_dir / "spread_table.csv")
    plot_estimate_spread(spread_by_noise, chance_spread(candidate_params, cf_list_hz),
                         run_dir / "estimate_spread.png", n_voxels, n_repeats)
    logger.info("done -> %s", run_dir.resolve())
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    grid = commands.add_parser("build-grid", help="candidate grid of ONE run order (un-normalised)")
    grid.add_argument("--design-seed", type=int, required=True)
    grid.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    grid.add_argument("--output-dir", default=DEFAULT_GRID_DIR)
    grid.add_argument("--n-workers", type=int, default=n_workers_default)

    recover = commands.add_parser("recover", help="24-run recovery (joint fit and random effects)")
    recover.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    recover.add_argument("--grid-dir", default=DEFAULT_GRID_DIR)
    recover.add_argument("--baseline-dir", required=True,
                         help="folder with the single-run sweep_recovery_results_design1000_<noise>_*.npz")
    recover.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    recover.add_argument("--noise", nargs="+", default=["low", "mid", "high"],
                         choices=["low", "mid", "high"])
    recover.add_argument("--n-voxels", type=int, default=N_VOXELS)
    recover.add_argument("--n-repeats", type=int, default=N_REPEATS)
    recover.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    recover.add_argument("--n-workers", type=int, default=n_workers_default)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import stimuli.sweep_design_erb as design   # needs the project root on sys.path
    raw_psth_by_seq, cf_list_hz = load_raw_psths(Path(args.results_dir))
    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)

    if args.command == "build-grid":
        run_designs, total_run_dur_s = build_run_designs([args.design_seed])
        candidate_params, template_matrix = build_candidate_grid(
            raw_psth_by_seq, len(cf_list_hz), run_designs[0], total_run_dur_s, design.TR_S,
            design.SILENCE_SEQ_ID, hrf_kernel, n_workers=args.n_workers, normalize=False)
        save_run_grid(Path(args.output_dir) / f"run_grid_design{args.design_seed}.npz",
                      candidate_params, template_matrix)
    else:
        run_designs, total_run_dur_s = build_run_designs(args.design_seeds)
        candidate_params, template_per_run = load_run_grids(Path(args.grid_dir), args.design_seeds)
        recover_and_report(
            raw_psth_by_seq, cf_list_hz, run_designs, total_run_dur_s, design.TR_S,
            design.SILENCE_SEQ_ID, hrf_kernel, candidate_params, template_per_run,
            Path(args.baseline_dir), Path(args.output_dir), list(dict.fromkeys(args.noise)),
            args.n_voxels, args.n_repeats, args.design_seeds, args.n_workers)


if __name__ == "__main__":
    main()
