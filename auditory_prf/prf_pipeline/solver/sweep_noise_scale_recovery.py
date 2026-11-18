"""sweep_noise_scale_recovery.py
=============================
At which noise level does the perfect recovery turn into the 'regression to the mean'?

The ground truth is the CLEAN signal of each voxel (as in the perfect noise-free recovery), for the 100
on-grid voxels of the earlier experiments and 24 run orders. Noise = k x the 'low' preset (white, cardiac,
respiratory and drift components together, jitter 0.1, added as is to the signal scaled by the global best-case
constant, like the earlier recoveries). k = 0 is the noise-free case (a perfect diagonal), k = 1 is the 'low'
noise. The noise of every level is exactly k times the SAME noise pattern (same seeds), so the levels differ
only in size.

At each level: run 1 alone, the 24-run joint fit and the 24-run random effects (log-space average), from
identical noise, with the tau true-versus-recovered heatmaps and the tau summary (share of exact recoveries,
correlation with the true tau, relative MSE). The clean signals are read from the 24 template grids of
``sweep_recovery_multirun.py build-grid`` (verified against the forward model), so no cochlear results are
needed for the signals, only the CF list.

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

from auditory_prf.prf_pipeline.run_assembly_sweeps import load_raw_psths
from auditory_prf.prf_pipeline.solver.additive_noise import SIGNAL_SCALE
from auditory_prf.prf_pipeline.solver.grid_recovery import MC_SEED
from auditory_prf.prf_pipeline.solver.parallel_recovery import REPEAT_CHUNK, map_voxels
from auditory_prf.prf_pipeline.solver.sweep_highpass_drift_test import DEFAULT_CUTOFFS_S, highpass_projection
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    PARAM_NAMES,
    clean_margins,
    plot_recovered_level_distribution,
    sample_ground_truths,
    save_summary_table,
    summarize,
)
from auditory_prf.prf_pipeline.solver.sweep_recovery_multirun import (
    DEFAULT_GRID_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    ESTIMATES,
    METHOD_COLORS,
    METHOD_LABELS,
    RUN_DESIGN_SEEDS,
    average_run_estimates,
    concatenate_templates,
    draw_multirun_measurement,
    load_run_grids,
    normalize_run_templates,
)
from prf_models.pm_noise import PmAdapter, PmNoise

logger = logging.getLogger(__name__)

NOISE_SCALES = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
NOISE_JITTER_SWEEP = [0.1, 0.1, 0.1]            # same jitter as the averaged-ground-truth experiment
N_REPEATS_SWEEP = 200
TOP_N_MATCHES = 10                              # best candidates kept for every matching
N_VOXELS = 100
TAU_INDEX = PARAM_NAMES.index("tau_ms")


# ── Size of the noise ─────────────────────────────────────────────────────────
def total_noise_sd_percent(preset: str, n_tr: int, tr_s: float, jitter: Sequence[float] = NOISE_JITTER_SWEEP,
                           n_seeds: int = 200) -> float:
    """SD of the full noise of a preset at k = 1 (white + cardiac + respiratory + drift), in % of the mean signal.

    Root-mean-square over ``n_seeds`` noise draws (seeds 1 .. n_seeds) of the standard deviation over time of one
    draw. The noise is linear in k, so the SD at level k is ``k`` times this value.

    Parameters
    ----------
    n_tr : int
        Number of volumes of one run.
    tr_s : float
        Repetition time in seconds.

    Returns
    -------
    float
    """
    variances = []
    for seed in range(1, n_seeds + 1):
        noise_model = PmNoise(voxel=preset, seed=seed, jitter=list(jitter))
        noise_model.pm = PmAdapter(TR=tr_s, time_points_n=n_tr, time_points_series=np.arange(n_tr) * tr_s)
        noise_model.compute()
        variances.append(np.var(noise_model.values_array))
    return float(100.0 * np.sqrt(np.mean(variances)))


# ── Clean signals ─────────────────────────────────────────────────────────────
def clean_signals_from_grids(candidate_params: np.ndarray, template_per_run: list,
                             true_params: np.ndarray) -> np.ndarray:
    """Clean signal of every voxel in every run: the template row of its true candidate.

    Returns
    -------
    np.ndarray float64, shape (n_voxels, n_runs, n_tr), the model units of the grids (mean-removed)
    """
    rows = []
    for true_row in true_params:
        match = np.flatnonzero(np.all(np.isclose(candidate_params, true_row), axis=1))
        assert len(match) == 1, f"{true_row} is not one candidate of the grid"
        rows.append(match[0])
    return np.stack([np.stack([templates[row].astype(np.float64) for templates in template_per_run])
                     for row in rows])


def level_indices_from_params(candidate_params: np.ndarray) -> list:
    """For every parameter, the level number (0 ..) of each candidate.

    Returns
    -------
    list of 4 np.ndarray int64, each (n_candidates,); ``np.unique`` order, so level j of a parameter is its j-th
    smallest value.
    """
    return [np.unique(candidate_params[:, p], return_inverse=True)[1].astype(np.int64).ravel()
            for p in range(candidate_params.shape[1])]


def match_details(template_matrix_norm: np.ndarray, measurements: np.ndarray, level_indices: list,
                  top_n: int = TOP_N_MATCHES, repeat_chunk: int = REPEAT_CHUNK) -> dict:
    """Pearson matching like ``best_match_indices``, but keeping the best candidates and the level profiles.

    The winner is identical to ``best_match_indices`` (same normalisation, same matrix product); the input is
    not modified.

    Parameters
    ----------
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
        Mean-removed, unit-norm templates.
    measurements : np.ndarray, shape (n_repeats, n_timepoints)
    level_indices : list of np.ndarray int, each (n_candidates,)
        ``level_indices_from_params``.
    top_n : int
        Number of best candidates kept per measurement (at most the number of candidates).

    Returns
    -------
    dict with
        best_idx : (n_repeats,) int64, the winner (the highest Pearson r, lowest index on a tie)
        top_idx : (n_repeats, top_n) int16, the best candidates, best first
        top_r : (n_repeats, top_n) float32, their Pearson r
        level_best_r : (n_repeats, n_levels_total) float32, the highest r among the candidates of every level of
            every parameter (the parameters' levels one after the other)
    """
    n_candidates = template_matrix_norm.shape[0]
    assert measurements.shape[1] == template_matrix_norm.shape[1], "timepoints differ"
    assert n_candidates < np.iinfo(np.int16).max, "candidate numbers do not fit int16"
    top_n = min(top_n, n_candidates)
    centered = measurements - measurements.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.where(norms == 0, 1.0, norms)
    groups = []
    for level_index in level_indices:                      # candidates sorted by level, start of every level
        order = np.argsort(level_index, kind="stable")
        groups.append((order, np.flatnonzero(np.r_[True, np.diff(level_index[order]) != 0])))
    n_repeats = measurements.shape[0]
    best_idx = np.empty(n_repeats, dtype=np.int64)
    top_idx = np.empty((n_repeats, top_n), dtype=np.int16)
    top_r = np.empty((n_repeats, top_n), dtype=np.float32)
    level_best_r = np.empty((n_repeats, sum(len(starts) for _, starts in groups)), dtype=np.float32)
    for start in range(0, n_repeats, repeat_chunk):
        stop = min(start + repeat_chunk, n_repeats)
        scores = template_matrix_norm @ unit[start:stop].T                    # (n_candidates, chunk)
        best_idx[start:stop] = np.argmax(scores, axis=0)
        if top_n < n_candidates:
            candidates = np.argpartition(scores, n_candidates - top_n, axis=0)[n_candidates - top_n:]
        else:
            candidates = np.broadcast_to(np.arange(n_candidates)[:, None], scores.shape)
        candidate_r = np.take_along_axis(scores, candidates, axis=0)
        order = np.lexsort((candidates, -candidate_r), axis=0)               # best r first, lowest index on a tie
        top_idx[start:stop] = np.take_along_axis(candidates, order, axis=0).T
        top_r[start:stop] = np.take_along_axis(candidate_r, order, axis=0).T
        offset = 0
        for sort_order, starts in groups:
            level_best_r[start:stop, offset:offset + len(starts)] = np.maximum.reduceat(
                scores[sort_order], starts, axis=0).T
            offset += len(starts)
    return {"best_idx": best_idx, "top_idx": top_idx, "top_r": top_r, "level_best_r": level_best_r}


def highpass_template_per_run(template_per_run: list, projection: np.ndarray) -> list:
    """High-pass every run's templates with the same projection; the amplitudes are kept (the joint fit needs them).

    Parameters
    ----------
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)
    projection : np.ndarray, shape (n_tr, n_tr)
        ``sweep_highpass_drift_test.highpass_projection``; it also removes the mean.

    Returns
    -------
    list of np.ndarray float32, each (n_candidates, n_tr); pass it to ``concatenate_templates`` and
    ``normalize_run_templates`` like the unfiltered templates.
    """
    return [(templates.astype(np.float64) @ projection).astype(np.float32) for templates in template_per_run]


# ── Recovery ──────────────────────────────────────────────────────────────────
def _voxel_task(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: all noise repeats of one voxel at one noise scale; joint and per-run argmax, margins.

    With ``state["projection"]`` the slow components of every run are removed from the measurement (the
    templates in ``state`` must be filtered with the same projection).
    """
    clean_per_run = list(state["clean_signals"][v_i])                              # n_runs arrays (n_tr,)
    n_runs, n_tr, n_repeats = len(clean_per_run), len(clean_per_run[0]), state["n_repeats"]
    measurements = np.empty((n_repeats, n_runs * n_tr), dtype=np.float32)
    for rep in range(n_repeats):
        measurements[rep] = draw_multirun_measurement(
            clean_per_run, state["preset"], state["tr_s"], v_i, rep, state["jitter"],
            signal_scale=state["signal_scale"], noise_scale=state["noise_scale"])
    raw_measurements = measurements.copy() if state["save_measurements"] else None   # before any filter
    projection = state["projection"]
    if projection is not None:                       # same noise draws, slow components of each run removed
        measurements = (measurements.reshape(n_repeats, n_runs, n_tr) @ projection.astype(np.float32)
                        ).reshape(n_repeats, n_runs * n_tr)
        clean_per_run = [bold @ projection for bold in clean_per_run]
    per_run = [match_details(state["template_per_run_norm"][r], measurements[:, r * n_tr:(r + 1) * n_tr],
                             state["level_indices"], state["top_n"]) for r in range(n_runs)]
    joint = match_details(state["template_matrix_norm"], measurements, state["level_indices"], state["top_n"])
    clean_concatenated = np.concatenate([bold - bold.mean() for bold in clean_per_run])
    result = clean_margins(clean_concatenated, state["template_matrix_norm"], state["candidate_params"])
    result["recovered"] = state["candidate_params"][joint["best_idx"]]
    result["recovered_per_run"] = state["candidate_params"][np.stack([d["best_idx"] for d in per_run], axis=1)]
    for name in ("top_idx", "top_r", "level_best_r"):                    # every match, run by run: (n_repeats, n_runs, .)
        result[f"match_{name}_per_run"] = np.stack([d[name] for d in per_run], axis=1)
        result[f"match_{name}_joint"] = joint[name]                      # (n_repeats, .)
    result["measurements"] = raw_measurements
    return result


def _recover_at_scale(clean_signals, candidate_params, template_matrix_norm, template_per_run_norm, true_params,
                      preset, noise_scale, n_repeats, tr_s, cf_list_hz, n_workers, jitter, signal_scale,
                      projection, top_n, save_measurements) -> dict:
    state = dict(clean_signals=clean_signals, candidate_params=candidate_params,
                 template_matrix_norm=template_matrix_norm, template_per_run_norm=template_per_run_norm,
                 preset=preset, noise_scale=noise_scale, n_repeats=n_repeats, tr_s=tr_s, jitter=jitter,
                 signal_scale=signal_scale, projection=projection, top_n=top_n,
                 save_measurements=save_measurements, level_indices=level_indices_from_params(candidate_params))
    if n_workers > 1:
        voxel_results = list(map_voxels(_voxel_task, true_params, state, n_workers))
    else:
        voxel_results = [_voxel_task(v_i, row, state) for v_i, row in enumerate(true_params)]
    recovered_per_run = np.stack([r["recovered_per_run"] for r in voxel_results])
    match_details_stacked = {key: np.stack([r[key] for r in voxel_results]) for key in voxel_results[0]
                             if key.startswith("match_")}
    return {
        **match_details_stacked,           # match_{top_idx,top_r,level_best_r}_{per_run,joint}: (n_voxels, n_repeats, ...)
        "measurements": np.stack([r["measurements"] for r in voxel_results]) if save_measurements else None,
        "recovered_params": np.stack([r["recovered"] for r in voxel_results]),         # joint fit
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


def run_noise_scale_sweep(
    clean_signals: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    template_per_run_norm: list,
    true_params: np.ndarray,
    preset: str,
    noise_scales: Sequence[float],
    n_repeats: int,
    tr_s: float,
    cf_list_hz: np.ndarray,
    n_workers: int = 1,
    jitter: list = NOISE_JITTER_SWEEP,
    signal_scale: float = SIGNAL_SCALE,
    projection: Optional[np.ndarray] = None,
    top_n: int = TOP_N_MATCHES,
    save_measurements: bool = False,
) -> dict:
    """Recovery from the clean signals with noise = k x the preset, for every k, on the same voxels and seeds.

    Parameters
    ----------
    clean_signals : np.ndarray, shape (n_voxels, n_runs, n_tr)    (``clean_signals_from_grids``)
    template_matrix_norm : np.ndarray, shape (n_candidates, n_runs * n_tr)   (joint fit)
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)    (one fit per run)
    noise_scales : sequence of float
        k = 0 is the noise-free case, 1 the preset unchanged.
    projection : np.ndarray, shape (n_tr, n_tr), optional
        High-pass projection (``sweep_highpass_drift_test.highpass_projection``) applied to every run of the
        noisy measurement. The templates must already be filtered with it (``highpass_template_per_run``).
        None = no filter.
    top_n : int
        Best candidates kept for every matching (``match_details``).
    save_measurements : bool
        Also return the noisy measurements (before any filter) under ``"measurements"``,
        (n_voxels, n_repeats, n_runs * n_tr) float32; otherwise that entry is None.

    Returns
    -------
    dict {k: result} with the keys of ``sweep_recovery_multirun.run_recovery_multirun``: recovered_params
    (n_voxels, n_repeats, 4; joint), recovered_per_run (n_voxels, n_repeats, n_runs, 4),
    recovered_random_effects and recovered_first_run (n_voxels, n_repeats, 4), margins, n_ties, ...
    """
    n_voxels, n_runs, n_tr = clean_signals.shape
    assert n_voxels == len(true_params) and n_runs == len(template_per_run_norm), "voxels or runs differ"
    assert template_per_run_norm[0].shape[1] == n_tr and template_matrix_norm.shape[1] == n_runs * n_tr
    results = {}
    for noise_scale in noise_scales:
        logger.info("noise scale k=%g x '%s': %d voxels x %d repeats x %d runs ...", noise_scale, preset,
                    n_voxels, n_repeats, n_runs)
        results[float(noise_scale)] = _recover_at_scale(
            clean_signals, candidate_params, template_matrix_norm, template_per_run_norm, true_params, preset,
            float(noise_scale), n_repeats, tr_s, cf_list_hz, n_workers, jitter, signal_scale, projection,
            top_n, save_measurements)
    return results


# ── Summary ───────────────────────────────────────────────────────────────────
def tau_summary(true_tau_ms: np.ndarray, recovered_tau_ms: np.ndarray, levels_ms: np.ndarray) -> dict:
    """Three numbers for tau.

    Parameters
    ----------
    true_tau_ms : np.ndarray, shape (n_voxels,)
    recovered_tau_ms : np.ndarray, shape (n_voxels, n_estimates)
    levels_ms : np.ndarray
        Grid levels; a recovered value counts as exact when its nearest level (log distance) is the true one.

    Returns
    -------
    dict with ``exact_share`` (share of estimates on the true level), ``correlation`` (Pearson r between
    ln true tau and the voxel's mean ln recovered tau; NaN if one of them is constant), ``relative_mse``
    (mean ((recovered - true) / true)^2 over all estimates).
    """
    nearest = levels_ms[np.abs(np.log(recovered_tau_ms)[..., None] - np.log(levels_ms)).argmin(axis=-1)]
    mean_log = np.log(recovered_tau_ms).mean(axis=1)
    log_true = np.log(true_tau_ms)
    correlation = (float(np.corrcoef(log_true, mean_log)[0, 1])
                   if np.std(log_true) > 0 and np.std(mean_log) > 1e-12 else float("nan"))
    return {"exact_share": float((nearest == true_tau_ms[:, None]).mean()),
            "correlation": correlation,
            "relative_mse": float((((recovered_tau_ms - true_tau_ms[:, None]) / true_tau_ms[:, None]) ** 2).mean())}


def best_constant_relative_mse(true_tau_ms: np.ndarray) -> float:
    """Relative MSE of the best single value for every voxel: c* = mean(1/t) / mean(1/t^2)."""
    c = np.mean(1.0 / true_tau_ms) / np.mean(1.0 / true_tau_ms ** 2)
    return float((((c - true_tau_ms) / true_tau_ms) ** 2).mean())


def save_tau_summary_table(rows: list, save_path: Path) -> None:
    """CSV: one row per filter x noise scale x method."""
    with open(save_path, "w") as f:
        f.write("noise_scale,total_sd_percent,white_sd_percent,filter,method,tau_exact_share,tau_correlation,"
                "tau_relative_mse\n")
        for row in rows:
            f.write(f"{row['noise_scale']:g},{row['total_sd_percent']:.4g},{row['white_sd_percent']:.4g},"
                    f"{row['filter']},{row['method']},{row['exact_share']:.6g},{row['correlation']:.6g},"
                    f"{row['relative_mse']:.6g}\n")


FILTER_LINESTYLES = ["-", "--", ":", "-."]


def plot_tau_summary(rows: list, noise_scales: Sequence[float], n_levels: int, best_constant: float,
                     save_path: Path, preset: str, n_voxels: int, n_repeats: int) -> None:
    """Three panels against the noise scale (evenly spaced, 0 on the left): exact share, correlation, relative MSE.

    One line per method (colour) and filter (line style: solid = no filter).
    """
    x = np.arange(len(noise_scales))
    filters = list(dict.fromkeys(r["filter"] for r in rows))
    labels = [f"{k:g}\n({next(r['total_sd_percent'] for r in rows if r['noise_scale'] == k):.2g} %)"
              for k in noise_scales]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = (("exact_share", "share of exact tau recoveries", 1.0 / n_levels, "uniform guess"),
              ("correlation", "correlation: true ln tau vs recovered", 0.0, "no relation"),
              ("relative_mse", "relative MSE of tau", best_constant, "best constant guess"))
    for ax, (key, title, reference, reference_label) in zip(axes, panels):
        for filter_i, filter_name in enumerate(filters):
            for method in METHOD_LABELS:
                values = [next(r[key] for r in rows if r["noise_scale"] == k and r["method"] == method
                               and r["filter"] == filter_name) for k in noise_scales]
                label = METHOD_LABELS[method]
                if len(filters) > 1:
                    label += ", " + ("no filter" if filter_name == "none" else filter_name)
                ax.plot(x, values, marker="o", ls=FILTER_LINESTYLES[filter_i % len(FILTER_LINESTYLES)],
                        color=METHOD_COLORS[method], label=label)
        ax.axhline(reference, color="k", ls="--", lw=0.9, label=reference_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel(f"noise scale k x '{preset}' (total noise SD in % of the mean signal)")
        ax.set_title(title, fontsize=10)
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Tau recovery against the noise level: clean ground truth, {n_voxels} voxels, "
                 f"{n_repeats} repeats, 24 runs", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Report and command line ───────────────────────────────────────────────────
def _report_one_filter(filter_name: str, results: dict, true_params: np.ndarray, candidate_params: np.ndarray,
                       cf_list_hz: np.ndarray, run_dir: Path, timestamp: str, preset: str, n_repeats: int,
                       white_sd_percent: dict, total_sd_percent: dict) -> list:
    """Tables, result files and heatmaps of one filter ("none" keeps the original file names); returns the summary rows."""
    infix = "" if filter_name == "none" else f"{filter_name}_"
    suffix = "" if filter_name == "none" else f"_{filter_name}"
    levels_ms = np.unique(candidate_params[:, TAU_INDEX])
    true_tau = true_params[:, TAU_INDEX]
    level_values = {f"level_values_{name}": np.unique(candidate_params[:, i]) for i, name in enumerate(PARAM_NAMES)}
    rows, recovered_by_method = [], {"1run": {}, "joint": {}, "random_effects": {}}
    for k, result in results.items():
        recovered = {"1run": result["recovered_first_run"], "joint": result["recovered_params"],
                     "random_effects": result["recovered_random_effects"]}
        for method_key, estimate in (("1run", "1run"), ("24run_joint", "joint"),
                                     ("24run_random_effects", "random_effects")):
            tau = recovered[estimate][:, :, TAU_INDEX]
            rows.append({"noise_scale": k, "total_sd_percent": total_sd_percent[k],
                         "white_sd_percent": white_sd_percent[k], "filter": filter_name, "method": method_key,
                         **tau_summary(true_tau, tau, levels_ms)})
            logger.info("%s k=%g %s: tau exact %.3f correlation %.3f relative MSE %.3f", filter_name, k,
                        method_key, rows[-1]["exact_share"], rows[-1]["correlation"], rows[-1]["relative_mse"])
        label = f"k={k:g}"
        for estimate in recovered_by_method:
            recovered_by_method[estimate][label] = recovered[estimate]
        for estimate in ESTIMATES:                              # the usual all-parameter summaries
            summary = summarize(true_params, {**result, "recovered_params": recovered[estimate]},
                                candidate_params, cf_list_hz)
            save_summary_table(summary, run_dir / f"summary_table_{estimate}_{infix}k{k:g}.csv")
        np.savez_compressed(
            run_dir / f"noise_scale_recovery_{infix}k{k:g}_{timestamp}.npz", true_params=true_params,
            recovered_params=result["recovered_params"], recovered_per_run=result["recovered_per_run"],
            recovered_random_effects=result["recovered_random_effects"], margins=result["margins"],
            n_ties=result["n_ties"], noise_scale=k, preset=preset, jitter=np.asarray(NOISE_JITTER_SWEEP),
            param_names=PARAM_NAMES, signal_scale=SIGNAL_SCALE, white_sd_percent=white_sd_percent[k],
            total_sd_percent=total_sd_percent[k], filter=filter_name, candidate_params=candidate_params,
            **level_values, **{key: value for key, value in result.items() if key.startswith("match_")})
    for method, recovered_by_label in recovered_by_method.items():
        plot_recovered_level_distribution(
            true_params, recovered_by_label, candidate_params,
            run_dir / f"recovered_tau_distribution_{method}{suffix}.png", param_name="tau_ms", n_repeats=n_repeats)
    return rows


def sweep_and_report(
    grid_dir: Path,
    output_dir: Path,
    cf_list_hz: np.ndarray,
    preset: str,
    noise_scales: Sequence[float],
    n_voxels: int,
    n_repeats: int,
    design_seeds: Sequence[int],
    n_workers: int = 1,
    tr_s: Optional[float] = None,
    highpass_cutoffs_s: Sequence[float] = (),
    top_n: int = TOP_N_MATCHES,
    save_measurements: bool = False,
) -> Path:
    """Run the sweep and write the tables, plots and result files.

    Every noise draw is matched without a filter and once per high-pass cutoff (identical seeds, so the
    differences are due to the filter). Files of the unfiltered recovery keep their names; the others get the
    filter name ``hp<cutoff>``.

    Parameters
    ----------
    highpass_cutoffs_s : sequence of float
        High-pass cutoffs in seconds; empty = only the unfiltered recovery.
    top_n : int
        Best candidates saved for every matching, with the best correlation at every parameter level.
    save_measurements : bool
        Also save the noisy measurements (shared by all filters) as ``measurements_k<k>_<timestamp>.npy``,
        (n_voxels, n_repeats, n_runs * n_tr) float32 per noise level.

    Returns
    -------
    Path of the result folder ``<output_dir>/<timestamp>_noise_scale_<preset>_<n>runs``.
    """
    if tr_s is None:
        import stimuli.sweep_design_erb as design      # needs the project root on sys.path
        tr_s = design.TR_S
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(output_dir) / f"{timestamp}_noise_scale_{preset}_{len(design_seeds)}runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_params, template_per_run = load_run_grids(Path(grid_dir), design_seeds)
    true_params = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(cf_list_hz),
                                       candidate_params, n_voxels)
    clean_signals = clean_signals_from_grids(candidate_params, template_per_run, true_params)
    n_tr = clean_signals.shape[2]
    white_sd_percent = {k: 100.0 * k * PmNoise.defaults_get(preset)["white_amplitude"] for k in noise_scales}
    total_sd_at_k1_percent = total_noise_sd_percent(preset, n_tr, tr_s)
    total_sd_percent = {k: k * total_sd_at_k1_percent for k in noise_scales}

    rows = []
    filters = [("none", None)] + [(f"hp{cutoff:g}", cutoff) for cutoff in highpass_cutoffs_s]
    for filter_name, cutoff in filters:
        projection = None if cutoff is None else highpass_projection(n_tr, tr_s, cutoff)
        filtered = template_per_run if projection is None else highpass_template_per_run(template_per_run, projection)
        results = run_noise_scale_sweep(
            clean_signals, candidate_params, concatenate_templates(filtered), normalize_run_templates(filtered),
            true_params, preset, noise_scales, n_repeats, tr_s, cf_list_hz, n_workers, projection=projection,
            top_n=top_n, save_measurements=save_measurements and cutoff is None)
        for k, result in results.items():                  # the measurements are the same for every filter: save once
            if result["measurements"] is not None:
                np.save(run_dir / f"measurements_k{k:g}_{timestamp}.npy", result["measurements"])
                result["measurements"] = None
        rows += _report_one_filter(filter_name, results, true_params, candidate_params, cf_list_hz, run_dir,
                                   timestamp, preset, n_repeats, white_sd_percent, total_sd_percent)
    save_tau_summary_table(rows, run_dir / "tau_summary_by_noise_scale.csv")
    plot_tau_summary(rows, list(white_sd_percent), len(np.unique(candidate_params[:, TAU_INDEX])),
                     best_constant_relative_mse(true_params[:, TAU_INDEX]),
                     run_dir / "tau_summary_by_noise_scale.png", preset, n_voxels, n_repeats)
    logger.info("done -> %s", run_dir.resolve())
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="only used for the CF list")
    parser.add_argument("--grid-dir", default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset", choices=["low", "mid", "high"], default="low",
                        help="the noise preset that the scale k multiplies")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=list(NOISE_SCALES),
                        help="k values; 0 = noise-free, 1 = the preset")
    parser.add_argument("--n-voxels", type=int, default=N_VOXELS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS_SWEEP)
    parser.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    parser.add_argument("--highpass-cutoffs", type=float, nargs="*", default=list(DEFAULT_CUTOFFS_S),
                        help="high-pass cutoffs in seconds, each matched next to the unfiltered recovery on the "
                             "same noise; give the option without values for no filter")
    parser.add_argument("--top-n", type=int, default=TOP_N_MATCHES,
                        help="best candidates saved for every matching, with the best r at every parameter level")
    parser.add_argument("--save-measurements", action="store_true",
                        help="save the noisy measurements (about 0.8 GB per noise level for 100 voxels x 200 repeats)")
    parser.add_argument("--n-workers", type=int, default=n_workers_default)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, cf_list_hz = load_raw_psths(Path(args.results_dir))
    sweep_and_report(Path(args.grid_dir), Path(args.output_dir), cf_list_hz, args.preset,
                     list(dict.fromkeys(args.noise_scales)), args.n_voxels, args.n_repeats,
                     args.design_seeds, args.n_workers, highpass_cutoffs_s=args.highpass_cutoffs,
                     top_n=args.top_n, save_measurements=args.save_measurements)


if __name__ == "__main__":
    main()
