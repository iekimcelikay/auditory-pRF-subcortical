"""sweep_recovery_multirun.py
===========================
Recovery of (cf_index, alpha, tau_ms, w) from SEVERAL runs with different stimulus orders
(the real experiment has 24), with a FIXED-EFFECTS estimate.

Every run is scored on its own: the Pearson r of the run's noisy measurement (independent noise per run, mean
removed) against all candidates' templates for that run order. The r^2 tables of the runs are averaged and the
candidate with the highest mean r^2 wins. Nothing is thrown away: the best score of every level of every parameter
(e.g. every tau), the top candidates and the per-run profiles are saved, and the full score vector over all
candidates for the first few repeats of every voxel. The argmax-r estimate of each run alone is kept for the 1-run
baseline and the spread.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from auditory_prf.prf_pipeline.hrf import SUBCORTICAL_PARAMS, build_hrf_kernel
from auditory_prf.prf_pipeline.run_assembly_sweeps import (
    SIGNAL_DT_S,
    load_raw_psths,
    run_sweep_forward_model,
)
from auditory_prf.prf_pipeline.solver.additive_noise import NOISE_MODELS, signal_scale_for
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import (
    MC_SEED,
    plot_recovery,
    save_recovery_table,
)
from auditory_prf.prf_pipeline.solver.highpass import (
    DEFAULT_HIGHPASS_CUTOFFS_S,
    check_cutoff,
    filter_name,
    highpass_projection,
    highpass_template_per_run,
    parse_cutoffs,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    plot_accuracy_comparison,
    save_accuracy_table,
)
from auditory_prf.prf_pipeline.solver.match_details import (
    TOP_N_MATCHES,
    level_indices_from_params,
    match_details,
    match_details_fixed_effects,
)
from auditory_prf.prf_pipeline.solver.parallel_recovery import map_voxels
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    N_VOXELS,
    N_TOP_CANDIDATES,
    NOISE_JITTER,
    PARAM_NAMES,
    RECTIFY,
    RHO,
    TIE_TOL,
    build_candidate_grid,
    clean_margins,
    plot_relative_mse,
    plot_relative_mse_by_true_value,
    plot_recovered_level_distribution,
    plot_relative_mse_distribution,
    sample_ground_truths,
    save_margin_table,
    save_summary_table,
    summarize,
)
from auditory_prf.visualization.design_label import titled
from stimuli.sweep_design_names import DEFAULT_DESIGN_NAME, DESIGN_NAMES

logger = logging.getLogger(__name__)

RUN_DESIGN_SEEDS = tuple(range(1000, 1024))      # 24 run orders, one per design seed
RUN_PREFIX = "sweep_recovery_24runs"             # result-file prefix (see replot_from_results.py)
N_REPEATS = 1000                                 # Monte Carlo repeats per noise level (one repeat = one experiment)
ESTIMATES = ("fixed_effects",)                   # the 24-run estimate (mean r^2 over runs, argmax)
# Folders follow the design name (stimuli/sweep_design_names.py): models_output/<design>/combined,
# sweep_recovery_multirun_grids/<design>, sweep_recovery_results/<design>.


# ── Run designs and per-run grids ─────────────────────────────────────────────
def build_run_designs(design_seeds: Sequence[int], design=None) -> tuple:
    """Run orders of the given design seeds.

    Parameters
    ----------
    design : stimuli.sweep_designs.SweepDesign, optional
        Default: the ERB 3-band design (sweep_design_erb3band).

    Returns
    -------
    run_designs : list of run designs (each a list of (soundfileid, onset_s))
    total_run_dur_s : float
        Common run length in seconds (asserted equal for every run).
    """
    if design is None:
        from stimuli.sweep_designs import get_design   # needs the project root on sys.path
        design = get_design()
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


# ── Per-run templates, clean margins, spread ──────────────────────────────────
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


def clean_margins_fixed_effects(clean_per_run: np.ndarray, template_per_run_norm: list,
                                candidate_params: np.ndarray, tie_tol: float = TIE_TOL) -> dict:
    """Identifiability of one voxel from its CLEAN BOLD on the fixed-effects score (mean r^2 over runs).

    Every run is scored on its own (clean signal and templates mean-removed and unit-norm within the run), then
    the r^2 values are averaged over runs; computed in float64.

    Parameters
    ----------
    clean_per_run : np.ndarray, shape (n_runs, n_tr)
        Clean BOLD of the voxel in every run.
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        ``normalize_run_templates`` output.
    candidate_params : np.ndarray, shape (n_candidates, n_params)

    Returns
    -------
    dict with best_idx, r_best (the winner's mean r^2), margins (n_params,; the winner's mean r^2 minus the best
    mean r^2 among the candidates that differ in that parameter), n_ties (candidates within ``tie_tol`` of the
    best), top_idx / top_r (N_TOP_CANDIDATES; mean r^2).
    """
    assert len(clean_per_run) == len(template_per_run_norm), "one clean signal per run"
    mean_r2 = np.zeros(template_per_run_norm[0].shape[0])
    for clean_bold, templates in zip(clean_per_run, template_per_run_norm):
        centered = np.asarray(clean_bold, dtype=np.float64) - np.mean(clean_bold)
        norm = np.linalg.norm(centered)
        unit = centered / (norm if norm > 0 else 1.0)
        mean_r2 += (templates.astype(np.float64) @ unit) ** 2
    mean_r2 /= len(clean_per_run)
    best_idx = int(np.argmax(mean_r2))
    margins = np.full(candidate_params.shape[1], np.nan)
    for p in range(candidate_params.shape[1]):
        differs = ~np.isclose(candidate_params[:, p], candidate_params[best_idx, p])
        if differs.any():
            margins[p] = mean_r2[best_idx] - mean_r2[differs].max()
    top_idx = np.argsort(-mean_r2, kind="stable")[:N_TOP_CANDIDATES]
    return {"best_idx": best_idx, "r_best": float(mean_r2[best_idx]), "margins": margins,
            "n_ties": int((mean_r2 >= mean_r2[best_idx] - tie_tol).sum()),
            "top_idx": top_idx, "top_r": mean_r2[top_idx]}


def _to_average_space(estimates: np.ndarray, cf_list_hz: np.ndarray) -> np.ndarray:
    """(..., 4) estimates -> natural log of CF in Hz, alpha and tau_ms; w unchanged."""
    assert estimates.shape[-1] == 4, estimates.shape
    cf_index = np.round(estimates[..., 0]).astype(int)
    assert cf_index.min() >= 0 and cf_index.max() < len(cf_list_hz), "cf_index outside the CF list"
    return np.stack([np.log(cf_list_hz[cf_index]), np.log(estimates[..., 1]),
                     np.log(estimates[..., 2]), estimates[..., 3]], axis=-1)


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


# ── 24-run recovery (fixed effects) ───────────────────────────────────────────
def draw_multirun_measurement(
    clean_bold_per_run: list,
    noise: str,
    tr_s: float,
    voxel_index: int,
    repeat_index: int,
    jitter: Optional[list] = NOISE_JITTER,
    signal_scale: Optional[float] = None,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """One noisy experiment: independent noise per run, each run mean-removed, concatenated.

    Run r of repeat k of voxel v gets the seed ``spawn_noise_seeds(MC_SEED, v, k, n_runs)[r]``;
    the seed of run 0 does not depend on ``n_runs`` (so run 0 equals the single-run noise).

    Parameters
    ----------
    clean_bold_per_run : list of np.ndarray, each (n_tr,)
    noise : str
        'low' | 'mid' | 'high', or 'none' for the clean BOLD.
    signal_scale : float or None
        None: legacy per-voxel noise scaling; a number: additive noise on a signal scaled by it
        (``additive_noise.SIGNAL_SCALE``).
    noise_scale : float
        Multiplier of the additive noise (1.0 = the preset); the same seeds at every scale, so the noise at
        scale k is exactly k times the noise at scale 1.

    Returns
    -------
    np.ndarray, shape (n_runs * n_tr,)
    """
    n_runs = len(clean_bold_per_run)
    assert len({len(bold) for bold in clean_bold_per_run}) == 1, "runs differ in length"
    run_seeds = spawn_noise_seeds(MC_SEED, voxel_index, repeat_index, n_runs=n_runs)
    noisy_per_run = draw_noisy_measurement(clean_bold_per_run, noise, tr_s, run_seeds,
                                           jitter=jitter, signal_scale=signal_scale,
                                           noise_scale=noise_scale).reshape(n_runs, -1)
    return (noisy_per_run - noisy_per_run.mean(axis=1, keepdims=True)).ravel()


def _recover_one_voxel_multirun(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: clean BOLD of every run, all noise repeats, fixed-effects and per-run matching, margins."""
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
            clean_bold_per_run, state["noise"], state["tr_s"], v_i, rep, state["jitter"],
            signal_scale=state.get("signal_scale"))
    measurements = measurements.reshape(n_repeats, n_runs, n_tr)
    projection = state.get("projection")
    if projection is not None:     # same noise draws, slow components of every run removed (templates filtered too)
        measurements = measurements @ projection.astype(np.float32)
        clean_bold_per_run = [bold @ projection for bold in clean_bold_per_run]

    level_indices, top_n = state["level_indices"], TOP_N_MATCHES
    fixed = match_details_fixed_effects(template_per_run_norm, measurements, level_indices, top_n,
                                        full_score_repeats=state["full_score_repeats"])
    per_run = [match_details(template_per_run_norm[r], measurements[:, r, :], level_indices, top_n)
               for r in range(n_runs)]                    # each run on its own: the single-run estimate (argmax r)

    result = clean_margins_fixed_effects(np.stack(clean_bold_per_run), template_per_run_norm, candidate_params)
    result["recovered"] = candidate_params[fixed["best_idx"]]                       # (n_repeats, 4)
    result["winner_mean_r"] = fixed["winner_mean_r"]                                # (n_repeats,)
    result["recovered_per_run"] = candidate_params[
        np.stack([d["best_idx"] for d in per_run], axis=1)]                         # (n_repeats, n_runs, 4)
    for name in ("top_idx", "top_r", "level_best_r"):     # every match: fixed effects and run by run
        result[f"match_{name}_fixed_effects"] = fixed[name]
        result[f"match_{name}_per_run"] = np.stack([d[name] for d in per_run], axis=1)
    if state["full_score_repeats"] > 0:
        result["full_scores_fixed_effects"] = fixed["full_scores"]                  # (n_full, n_candidates)
    return result


def run_recovery_multirun(
    raw_psth_by_seq: dict,
    run_designs: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    candidate_params: np.ndarray,
    template_per_run_norm: list,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    n_workers: int = 1,
    jitter: Optional[list] = NOISE_JITTER,
    signal_scale: Optional[float] = None,
    full_score_repeats: int = 0,
    projection: Optional[np.ndarray] = None,
) -> dict:
    """Monte Carlo recovery from all runs, one noise level: the fixed-effects estimate.

    Every run is scored on its own (Pearson r against all candidates), the r^2 values are averaged over the
    runs and the candidate with the highest mean r^2 wins.

    Parameters
    ----------
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        ``normalize_run_templates`` output (one matrix per run).
    full_score_repeats : int
        The mean-r^2 vector over all candidates is kept for the first this many repeats of every voxel (0 = none).
    projection : np.ndarray, shape (n_tr, n_tr), optional
        High-pass projection (``highpass.highpass_projection``) applied to every run of the noisy measurement and
        of the clean signal used for the margins. ``template_per_run_norm`` must then be made from templates
        filtered with the same projection (``highpass.highpass_template_per_run``). None = no filter.

    Returns
    -------
    dict with the keys of ``sweep_recovery.run_recovery`` for the FIXED-EFFECTS estimate
    (recovered_params (n_voxels, n_repeats, 4), best_idx, r_best, n_ties, margins, top_idx, top_r; the last
    five describe the clean ground truth on the mean-r^2 score), plus
        winner_mean_r (n_voxels, n_repeats): the winner's mean signed r (negative = anti-correlated),
        recovered_per_run (n_voxels, n_repeats, n_runs, 4): the argmax-r estimate of every run alone,
        recovered_first_run (n_voxels, n_repeats, 4): run 1 alone = the single-run estimate,
        match_{top_idx,top_r,level_best_r}_fixed_effects (n_voxels, n_repeats, ...) and
        match_{top_idx,top_r,level_best_r}_per_run (n_voxels, n_repeats, n_runs, ...): the best candidates and the
        best score of every level of every parameter,
        full_scores_fixed_effects (n_voxels, min(full_score_repeats, n_repeats), n_candidates), only if
        ``full_score_repeats > 0``.
    """
    assert len(run_designs) == len(template_per_run_norm), "one template matrix per run order"
    state = dict(raw_psth_by_seq=raw_psth_by_seq, run_designs=run_designs,
                 total_run_dur_s=total_run_dur_s, tr_s=tr_s, silence_seq_id=silence_seq_id,
                 hrf_kernel=hrf_kernel, candidate_params=candidate_params,
                 template_per_run_norm=template_per_run_norm, noise=noise, n_repeats=n_repeats,
                 jitter=jitter, signal_scale=signal_scale, full_score_repeats=full_score_repeats,
                 projection=projection, level_indices=level_indices_from_params(candidate_params))
    if n_workers > 1:
        voxel_results = list(map_voxels(_recover_one_voxel_multirun, true_params, state, n_workers))
    else:
        voxel_results = [_recover_one_voxel_multirun(v_i, row, state)
                         for v_i, row in enumerate(true_params)]
    recovered_per_run = np.stack([r["recovered_per_run"] for r in voxel_results])
    result = {
        "recovered_params": np.stack([r["recovered"] for r in voxel_results]),
        "winner_mean_r": np.stack([r["winner_mean_r"] for r in voxel_results]),
        "best_idx": np.array([r["best_idx"] for r in voxel_results]),
        "r_best": np.array([r["r_best"] for r in voxel_results]),
        "n_ties": np.array([r["n_ties"] for r in voxel_results]),
        "margins": np.stack([r["margins"] for r in voxel_results]),
        "top_idx": np.stack([r["top_idx"] for r in voxel_results]),
        "top_r": np.stack([r["top_r"] for r in voxel_results]),
        "recovered_per_run": recovered_per_run,
        "recovered_first_run": recovered_per_run[:, :, 0, :],
    }
    for key in voxel_results[0]:
        if key.startswith("match_") or key == "full_scores_fixed_effects":
            result[key] = np.stack([r[key] for r in voxel_results])
    return result


# ── Baseline, comparison and spread outputs ───────────────────────────────────
METHOD_LABELS = {"1run": "1 run", "24run_fixed_effects": "24 runs, fixed effects"}
METHOD_COLORS = {"1run": "gray", "24run_fixed_effects": "steelblue"}
NOISE_COLORS = {"none": "dimgray", "low": "forestgreen", "mid": "darkorange", "high": "firebrick"}
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
        Ground truths of the new run; asserted equal to the FIRST n_voxels voxels of the file
        (same voxels, same order). A smaller run (e.g. a smoke test) is allowed because the
        ground-truth draw of n voxels is the first n voxels of the draw of any larger number.

    Returns
    -------
    dict with recovered_params (n_voxels, n_repeats, 4), margins (n_voxels, 4), n_ties (n_voxels,) and
    noise_model ('additive' | 'legacy'; files without the key were made with the legacy noise)
    """
    paths = sorted(Path(baseline_dir).glob(f"sweep_recovery_results_design1000_{noise}_*.npz"))
    assert len(paths) == 1, f"expected one baseline file for '{noise}' in {baseline_dir}, found {paths}"
    with np.load(paths[0], allow_pickle=True) as baseline:
        assert baseline["recovered_params"].shape[1] >= n_repeats, (
            f"{paths[0].name} has {baseline['recovered_params'].shape[1]} repeats, need {n_repeats}")
        n_voxels = len(true_params)
        assert baseline["true_params"].shape[0] >= n_voxels, (
            f"{paths[0].name} has {baseline['true_params'].shape[0]} voxels, need {n_voxels}")
        assert np.allclose(baseline["true_params"][:n_voxels], true_params), (
            f"{paths[0].name}: other ground-truth voxels")
        noise_model = str(baseline["noise_model"]) if "noise_model" in baseline.files else "legacy"
        return {"recovered_params": baseline["recovered_params"][:n_voxels, :n_repeats, :],
                "margins": baseline["margins"][:n_voxels], "n_ties": baseline["n_ties"][:n_voxels],
                "noise_model": noise_model}


def save_run_comparison_table(summary_by_method: dict, save_path: Path) -> None:
    """Long CSV: one row per noise level x parameter x method.

    summary_by_method : {"1run": {noise: summary}, "24run_fixed_effects": {noise: summary}}
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
                        n_repeats: Optional[int] = None, design_label: Optional[str] = None) -> None:
    """Relative MSE of the methods side by side: one panel per parameter, one group per noise level,
    one bar per method (1 run, 24 runs with fixed effects); black tick = chance level; log y axis.
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
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=0,
                   label="relative MSE = 1 (error as large as the true value)" if ax is axes[0] else None)
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
    fig.suptitle(titled(f"Relative MSE: 1 run vs 24 runs{size}", design_label))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Run comparison plot saved -> %s", Path(save_path).resolve())


def _spread_legend_handles(noise_levels: list) -> tuple:
    """Legend of the spread plots: one patch per noise level, the chance line, the voxel dot."""
    handles = [Patch(facecolor="none", edgecolor=NOISE_COLORS.get(noise, "steelblue"), linewidth=2,
                     label=f"{noise} noise: box = quartiles, bar = median") for noise in noise_levels]
    handles.append(Line2D([], [], color="black", linewidth=2.2,
                          label="chance (SD of a uniform guess over the grid levels)"))
    handles.append(Line2D([], [], marker="o", linestyle="", color="gray", markersize=4,
                          label="one voxel (median over its repeats)"))
    return handles


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
                         n_voxels: Optional[int] = None, n_repeats: Optional[int] = None,
                         design_label: Optional[str] = None) -> None:
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
        ax.axhline(chance_spread_by_param[p], color="black", linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels(noise_levels)
        ax.set_xlim(-0.6, len(noise_levels) - 0.4)
        top = max(chance_spread_by_param[p],
                  max(np.median(s[..., p], axis=1).max() for s in spread_by_noise.values()))
        ax.set_ylim(0, 1.12 * top)                                  # headroom above the chance line
        ax.set_title(SPREAD_AXIS_LABELS[name])
    axes[0].set_ylabel("spread of the 24 per-run estimates (SD)")
    fig.legend(handles=_spread_legend_handles(noise_levels), loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.12))
    size = f" (N={n_voxels} voxels, {n_repeats:,} MC repeats each)" if n_voxels and n_repeats else ""
    fig.suptitle(titled(f"Width of the 24 per-run estimates{size}", design_label))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Estimate spread plot saved -> %s", Path(save_path).resolve())


def plot_spread_by_true_value(spread_by_noise: dict, true_params: np.ndarray,
                              chance_spread_by_param: np.ndarray, candidate_params: np.ndarray,
                              save_path: Path, param_name: str = "tau_ms",
                              n_repeats: Optional[int] = None, design_label: Optional[str] = None) -> None:
    """Spread of the 24 per-run estimates grouped by the TRUE value of one parameter (default tau_ms).

    One group per grid level, one box per noise level; every dot is a voxel with that true value
    (the median over its repeats of the standard deviation across the 24 runs, in the space of the
    averaging); black line = spread of a uniform guess over the grid levels.
    spread_by_noise : {noise: (n_voxels, n_repeats, 4)} from ``estimate_spread``.
    """
    p = PARAM_NAMES.index(param_name)
    levels = np.unique(candidate_params[:, p])
    true_value = true_params[:, p]
    noise_levels = list(spread_by_noise)
    offsets = (np.arange(len(noise_levels)) - (len(noise_levels) - 1) / 2) * 0.27
    rng = np.random.default_rng(0)                              # fixed jitter, reproducible figure

    fig, ax = plt.subplots(figsize=(1.3 * len(levels) + 5, 4.8))
    for g, level in enumerate(levels):
        in_group = np.isclose(true_value, level)
        for i, noise in enumerate(noise_levels):
            if not in_group.any():
                continue
            per_voxel = np.median(spread_by_noise[noise][in_group][:, :, p], axis=1)
            color = NOISE_COLORS.get(noise, "steelblue")
            ax.boxplot(per_voxel, positions=[g + offsets[i]], widths=0.22, showfliers=False,
                       manage_ticks=False, boxprops=dict(color=color), whiskerprops=dict(color=color),
                       capprops=dict(color=color), medianprops=dict(color=color, linewidth=2))
            ax.scatter(g + offsets[i] + rng.uniform(-0.07, 0.07, size=len(per_voxel)), per_voxel,
                       s=9, alpha=0.6, color=color, linewidths=0, zorder=3)
    ax.axhline(chance_spread_by_param[p], color="black", linewidth=2.2, zorder=4)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{level:g}\n(n={int(np.isclose(true_value, level).sum())})" for level in levels])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"true {param_name} (n = voxels with that true value)")
    ax.set_ylabel(f"spread of the 24 per-run estimates\n(SD, {SPREAD_AXIS_LABELS[param_name]})")
    ax.legend(handles=_spread_legend_handles(noise_levels), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8)
    size = f" ({n_repeats:,} MC repeats per voxel)" if n_repeats else ""
    ax.set_title(titled(f"Width of the 24 per-run {param_name} estimates by true {param_name}{size}", design_label))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spread by true %s plot saved -> %s", param_name, Path(save_path).resolve())


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
    baseline_dir: Optional[Path],
    output_dir: Path,
    noise_levels: list,
    n_voxels: int,
    n_repeats: int,
    design_seeds: Sequence[int],
    n_workers: int = 1,
    signal_scale: Optional[float] = None,
    full_score_repeats: int = 5,
    highpass_cutoff_s: Optional[float] = None,
    design_name: Optional[str] = None,
    design_label: Optional[str] = None,
) -> Path:
    """Run the 24-run recovery at every noise level and write all tables, plots and result files.

    ``design_name`` / ``design_label`` (stimuli/sweep_design_names.py) are stored in every result file and the label is
    shown in the title of every plot; None = neither (the output is as before).

    ``full_score_repeats``: the full mean-r^2 vector over all candidates is saved for the first this many repeats
    of every voxel (0 = none).

    ``signal_scale``: None = legacy per-voxel noise scaling; ``additive_noise.SIGNAL_SCALE`` = MATLAB-style
    additive noise. The single-run baseline has to be made with the same noise model.

    Ground truths: the same on-grid draw as ``sweep_recovery`` (``default_rng(MC_SEED)``), so the
    voxels equal the single-run results used as baseline.

    Noise level ``"none"`` is the noise-free test: every repeat is identical, so use
    ``n_repeats=1``; there is no single-run result file for it, so the "1-run" column is the
    estimate from run 1 alone in the same call, and ``baseline_dir`` may be None if every noise
    level is ``"none"``.

    ``highpass_cutoff_s``: None = the unfiltered recovery (unchanged). A number = the same recovery after
    removing the slow DCT components (period longer than the cutoff, seconds) from the measurement and the
    templates of every run, on the same noise seeds; the result folder gets the suffix ``_hp<cutoff>``, the npz
    files the key ``highpass_cutoff_s``, and the "1-run" column is run 1 alone of this call (the single-run
    baseline files are unfiltered), so ``baseline_dir`` may be None.

    Returns
    -------
    Path of the result folder ``<output_dir>/<timestamp>_<noises>_on_grid_<n>runs`` (``..._hp<cutoff>`` for a
    filtered pass).
    """
    if highpass_cutoff_s is not None:         # before anything is created or computed
        check_cutoff(template_per_run[0].shape[1], tr_s, highpass_cutoff_s)
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    filter_suffix = "" if highpass_cutoff_s is None else f"_{filter_name(highpass_cutoff_s)}"
    run_dir = (Path(output_dir) /
               f"{timestamp}_{'-'.join(noise_levels)}_on_grid_{len(design_seeds)}runs{filter_suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    first_seed = design_seeds[0]
    noise_model = "legacy" if signal_scale is None else "additive"
    logger.info("noise=%s (%s model) voxels=%d repeats=%d runs=%d workers=%d -> %s", noise_levels,
                noise_model, n_voxels, n_repeats, len(design_seeds), n_workers, run_dir)

    projection, filter_saved = None, {}
    if highpass_cutoff_s is not None:
        projection = highpass_projection(template_per_run[0].shape[1], tr_s, highpass_cutoff_s)
        template_per_run = highpass_template_per_run(template_per_run, projection)
        filter_saved = {"highpass_cutoff_s": highpass_cutoff_s}
    template_per_run_norm = normalize_run_templates(template_per_run)
    true_params = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(cf_list_hz),
                                       candidate_params, n_voxels)

    # load the baselines first: a mismatch should stop the job before the expensive recovery
    # (a filtered pass has none: the single-run baseline files are unfiltered)
    needs_baseline = [noise for noise in noise_levels if noise != "none" and highpass_cutoff_s is None]
    assert baseline_dir is not None or not needs_baseline, \
        f"baseline_dir is required for the noise levels {needs_baseline}"
    baselines = {noise: load_baseline_results(baseline_dir, noise, n_repeats, true_params)
                 for noise in needs_baseline}

    summary_by_method = {"1run": {}, "24run_fixed_effects": {}}
    recovered_by_method = {"1run": {}, "fixed_effects": {}}   # for the tau heatmaps
    estimate_summary = {estimate: {} for estimate in ESTIMATES}
    accuracy = {estimate: {} for estimate in ESTIMATES}
    spread_by_noise = {}
    for noise in noise_levels:
        logger.info("noise '%s': %d voxels x %d repeats x %d runs ...", noise, n_voxels, n_repeats,
                    len(design_seeds))
        result = run_recovery_multirun(
            raw_psth_by_seq, run_designs, total_run_dur_s, tr_s, silence_seq_id, hrf_kernel,
            candidate_params, template_per_run_norm, true_params, noise,
            n_repeats, n_workers, signal_scale=signal_scale, full_score_repeats=full_score_repeats,
            projection=projection)
        logger.info("[%s] fixed-effects winner has a negative mean r in %.4f of the repeats", noise,
                    float(np.mean(result["winner_mean_r"] < 0)))
        if noise == "none" or highpass_cutoff_s is not None:
            # no single-run result file (noise-free, or the baseline files are unfiltered): run 1 alone of this
            # call is the 1-run column
            baseline_result = {"recovered_params": result["recovered_first_run"],
                               "margins": result["margins"], "n_ties": result["n_ties"]}
        else:
            baseline = baselines[noise]
            if baseline["noise_model"] != noise_model:
                logger.warning("[%s] the single-run baseline was made with the '%s' noise, this run uses '%s': "
                               "re-run the single-run recovery with --noise-model %s first", noise,
                               baseline["noise_model"], noise_model, noise_model)
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
        recovered = {"fixed_effects": result["recovered_params"]}
        recovered_by_method["1run"][noise] = baseline_result["recovered_params"]
        for estimate in ESTIMATES:
            recovered_by_method[estimate][noise] = recovered[estimate]
        summaries = {"1run": summarize(true_params, baseline_result, candidate_params, cf_list_hz)}
        for estimate in ESTIMATES:     # margins and ties describe the clean ground truth on the mean-r^2 score
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
            saved = {k: v for k, v in result.items() if k != "recovered_params"}
            np.savez(run_dir / f"{RUN_PREFIX}_{estimate}_results_design{first_seed}_{noise}_{timestamp}.npz",
                     true_params=true_params, recovered_params=recovered[estimate],
                     param_names=PARAM_NAMES, noise_model=noise_model,
                     design=design_name or "", design_label=design_label or "", **filter_saved, **saved)
            plot_recovery(true_params, recovered[estimate],
                          run_dir / f"recovery_scatter_{estimate}_{noise}.png", param_names=PARAM_NAMES,
                          design_label=design_label)
            save_recovery_table(true_params, recovered[estimate],
                                run_dir / f"recovery_table_{estimate}_{noise}.csv", param_names=PARAM_NAMES)
            save_summary_table(summaries[f"24run_{estimate}"],
                               run_dir / f"summary_table_{estimate}_{noise}.csv")
        save_margin_table(true_params, result, run_dir / f"margin_table_{noise}.csv")

    for estimate in ESTIMATES:
        save_accuracy_table(accuracy[estimate], run_dir / f"accuracy_table_{estimate}.csv",
                            param_names=PARAM_NAMES)
        plot_accuracy_comparison(accuracy[estimate], run_dir / f"accuracy_comparison_{estimate}.png",
                                 param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats,
                                 design_label=design_label)
        plot_relative_mse(estimate_summary[estimate], run_dir / f"relative_mse_{estimate}.png",
                          param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats,
                          design_label=design_label)
        plot_relative_mse_distribution(
            estimate_summary[estimate], run_dir / f"relative_mse_distribution_{estimate}.png",
            param_names=PARAM_NAMES, n_voxels=n_voxels, n_repeats=n_repeats, design_label=design_label)
        plot_relative_mse_by_true_value(
            estimate_summary[estimate], true_params, candidate_params,
            run_dir / f"relative_mse_by_tau_{estimate}.png", param_name="tau_ms", n_repeats=n_repeats,
            design_label=design_label)
    for method, recovered_by_noise in recovered_by_method.items():      # true-vs-recovered tau heatmaps
        plot_recovered_level_distribution(
            true_params, recovered_by_noise, candidate_params,
            run_dir / f"recovered_tau_distribution_{method}.png", param_name="tau_ms",
            n_repeats=n_repeats, design_label=design_label)
    save_run_comparison_table(summary_by_method, run_dir / "run_comparison_table.csv")
    plot_run_comparison(summary_by_method, run_dir / "run_comparison.png", n_voxels, n_repeats,
                        design_label=design_label)
    save_spread_table(spread_by_noise, chance_spread(candidate_params, cf_list_hz),
                      run_dir / "spread_table.csv")
    plot_estimate_spread(spread_by_noise, chance_spread(candidate_params, cf_list_hz),
                         run_dir / "estimate_spread.png", n_voxels, n_repeats, design_label=design_label)
    plot_spread_by_true_value(spread_by_noise, true_params, chance_spread(candidate_params, cf_list_hz),
                              candidate_params, run_dir / "estimate_spread_by_tau.png",
                              param_name="tau_ms", n_repeats=n_repeats, design_label=design_label)
    logger.info("done -> %s", run_dir.resolve())
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    design_option = argparse.ArgumentParser(add_help=False)       # shared by both commands
    design_option.add_argument("--design", choices=DESIGN_NAMES, default=DEFAULT_DESIGN_NAME,
                               help="Sweep design (stimuli/sweep_design_names.py): picks the run orders, the folders "
                                    "and the label shown in the plots.")

    grid = commands.add_parser("build-grid", parents=[design_option],
                               help="candidate grid of ONE run order (un-normalised)")
    grid.add_argument("--design-seed", type=int, required=True)
    grid.add_argument("--results-dir", default=None,
                      help="Combined cochlear results. Default: models_output/<design>/combined.")
    grid.add_argument("--output-dir", default=None,
                      help="Default: sweep_recovery_multirun_grids/<design>.")
    grid.add_argument("--n-workers", type=int, default=n_workers_default)

    recover = commands.add_parser("recover", parents=[design_option],
                                  help="24-run recovery (fixed effects: mean r^2 over runs)")
    recover.add_argument("--results-dir", default=None,
                         help="Combined cochlear results. Default: models_output/<design>/combined.")
    recover.add_argument("--grid-dir", default=None,
                         help="Default: sweep_recovery_multirun_grids/<design>.")
    recover.add_argument("--baseline-dir", default=None,
                         help="folder with the single-run sweep_recovery_results_design1000_<noise>_*.npz "
                              "(required unless --noise is only 'none'); e.g. a folder under sweep_recovery_results/<design>/")
    recover.add_argument("--output-dir", default=None,
                         help="Default: sweep_recovery_results/<design>.")
    recover.add_argument("--noise", nargs="+", default=["low", "mid", "high"],
                         choices=["none", "low", "mid", "high"],
                         help="'none' = noise-free test (use --n-repeats 1)")
    recover.add_argument("--n-voxels", type=int, default=N_VOXELS)
    recover.add_argument("--n-repeats", type=int, default=N_REPEATS)
    recover.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    recover.add_argument("--n-workers", type=int, default=n_workers_default)
    recover.add_argument("--full-score-repeats", type=int, default=5,
                         help="repeats per voxel whose full r^2 vector over all candidates is saved (0 = none)")
    recover.add_argument("--noise-model", choices=list(NOISE_MODELS), default="additive",
                         help="additive: MATLAB-style noise added as is to the signal scaled by one global "
                              "best-case constant (solver/additive_noise.py); legacy: noise scaled by "
                              "std(bold) per voxel (the earlier results)")
    recover.add_argument("--highpass-cutoffs", type=float, nargs="*", default=list(DEFAULT_HIGHPASS_CUTOFFS_S),
                         help="high-pass cutoffs in seconds; each one is recovered in its own result folder "
                              "(suffix _hp<cutoff>) next to the unfiltered recovery, on the same noise; "
                              "give the option without values (or 0) for the unfiltered recovery only")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from stimuli.sweep_designs import get_design   # needs the project root on sys.path
    design = get_design(args.design)
    logger.info("design=%s (%s)", design.name, design.label)
    raw_psth_by_seq, cf_list_hz = load_raw_psths(Path(args.results_dir or design.combined_dir))
    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)

    if args.command == "build-grid":
        run_designs, total_run_dur_s = build_run_designs([args.design_seed], design)
        candidate_params, template_matrix = build_candidate_grid(
            raw_psth_by_seq, len(cf_list_hz), run_designs[0], total_run_dur_s, design.TR_S,
            design.SILENCE_SEQ_ID, hrf_kernel, n_workers=args.n_workers, normalize=False)
        save_run_grid(Path(args.output_dir or design.grid_dir) / f"run_grid_design{args.design_seed}.npz",
                      candidate_params, template_matrix)
    else:
        if args.baseline_dir is None and any(noise != "none" for noise in args.noise):
            raise SystemExit("--baseline-dir is required unless --noise is only 'none'")
        run_designs, total_run_dur_s = build_run_designs(args.design_seeds, design)
        candidate_params, template_per_run = load_run_grids(Path(args.grid_dir or design.grid_dir), args.design_seeds)
        for cutoff_s in [None, *parse_cutoffs(args.highpass_cutoffs)]:     # unfiltered first, then each filter
            recover_and_report(
                raw_psth_by_seq, cf_list_hz, run_designs, total_run_dur_s, design.TR_S,
                design.SILENCE_SEQ_ID, hrf_kernel, candidate_params, template_per_run,
                Path(args.baseline_dir) if args.baseline_dir else None,
                Path(args.output_dir or design.results_dir),
                list(dict.fromkeys(args.noise)),
                args.n_voxels, args.n_repeats, args.design_seeds, args.n_workers,
                signal_scale=signal_scale_for(args.noise_model), full_score_repeats=args.full_score_repeats,
                highpass_cutoff_s=cutoff_s, design_name=design.name, design_label=design.label)


if __name__ == "__main__":
    main()
