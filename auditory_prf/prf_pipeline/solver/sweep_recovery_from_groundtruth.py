"""sweep_recovery_from_groundtruth.py
=================================
Script 2 of the averaged-noisy ground-truth experiment (design:
auditory-prf-docs-all/groundtruth_average_recovery_design.md).

Recovery of (cf_index, alpha, tau_ms, w) from the AVERAGED ground-truth signal of
``sweep_groundtruth_distribution.py`` plus fresh noise (same PmNoise and jitter, separate seed
stream), from 24 runs: joint fit (concatenated runs) and random effects (one estimate per run, then
the log-space average), and the spread of the 24 per-run estimates. Each noise level is recovered
against its own averaged ground truth. Needs the 24 run grids of ``sweep_recovery_multirun.py``
(``build-grid``); reads only the small ``groundtruth_mean_<noise>.npz``, never the distribution.

Run from the project root.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import MC_SEED, plot_recovery, save_recovery_table
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    plot_accuracy_comparison,
    save_accuracy_table,
)
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels
from auditory_prf.prf_pipeline.solver.sweep_groundtruth_distribution import NOISE_JITTER_GROUNDTRUTH
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    PARAM_NAMES,
    clean_margins,
    plot_recovered_level_distribution,
    plot_relative_mse,
    plot_relative_mse_by_true_value,
    plot_relative_mse_distribution,
    sample_ground_truths,
    save_margin_table,
    save_summary_table,
    summarize,
)
from auditory_prf.prf_pipeline.solver.sweep_recovery_multirun import (
    DEFAULT_GRID_DIR,
    DEFAULT_OUTPUT_DIR,
    ESTIMATES,
    RUN_DESIGN_SEEDS,
    average_run_estimates,
    chance_spread,
    concatenate_templates,
    estimate_spread,
    load_run_grids,
    normalize_run_templates,
    plot_estimate_spread,
    plot_run_comparison,
    plot_spread_by_true_value,
    save_run_comparison_table,
    save_spread_table,
)

logger = logging.getLogger(__name__)

SEED_TAG_RECOVERY = 2                            # seed stream of the recovery noise (ground truth uses tag 1)
N_REPEATS = 1000                                 # Monte Carlo repeats per noise level


# ── Ground truth and measurement ──────────────────────────────────────────────
def load_groundtruth(groundtruth_dir: Path, noise: str) -> dict:
    """Averaged ground truth of one noise level (reads only ``groundtruth_mean_<noise>.npz``).

    Returns
    -------
    dict with ``groundtruth_bold`` (n_voxels, n_runs, n_tr) float32, ``true_params`` (n_voxels, 4),
    ``cf_list_hz`` (n_cf,), ``jitter`` (list of 3 floats), ``n_repeats`` (int), ``tr_s`` (float),
    ``design_seeds`` (list of int).
    """
    path = Path(groundtruth_dir) / f"groundtruth_mean_{noise}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no averaged ground truth for noise level '{noise}': {path} (the file is written last by "
            "sweep_groundtruth_distribution.py, so a missing file means that level did not finish)")
    with np.load(path) as saved:
        assert str(saved["noise"]) == noise, (
            f"{path.name} was built for noise level '{saved['noise']}', not '{noise}'")
        return {"groundtruth_bold": saved["groundtruth_bold"], "true_params": saved["true_params"],
                "cf_list_hz": saved["cf_list_hz"], "jitter": [float(x) for x in saved["jitter"]],
                "n_repeats": int(saved["n_repeats"]), "tr_s": float(saved["tr_s"]),
                "design_seeds": [int(x) for x in saved["design_seeds"]]}


def draw_recovery_measurement(
    groundtruth_per_run: np.ndarray,
    noise: str,
    tr_s: float,
    voxel_index: int,
    repeat_index: int,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
) -> np.ndarray:
    """One noisy experiment from the averaged ground truth: independent noise per run, each run
    mean-removed, concatenated.

    Run r gets the seed ``spawn_noise_seeds(MC_SEED, SEED_TAG_RECOVERY, voxel, repeat, n_runs)[r]``.

    Parameters
    ----------
    groundtruth_per_run : np.ndarray, shape (n_runs, n_tr)
    noise : str
        'low' | 'mid' | 'high', or 'none' for the ground truth without new noise.

    Returns
    -------
    np.ndarray, shape (n_runs * n_tr,)
    """
    n_runs = len(groundtruth_per_run)
    run_seeds = spawn_noise_seeds(MC_SEED, SEED_TAG_RECOVERY, voxel_index, repeat_index, n_runs=n_runs)
    noisy_per_run = draw_noisy_measurement(
        list(groundtruth_per_run), noise, tr_s, run_seeds, jitter=jitter).reshape(n_runs, -1)
    return (noisy_per_run - noisy_per_run.mean(axis=1, keepdims=True)).ravel()


# ── Recovery ──────────────────────────────────────────────────────────────────
def _recover_one_voxel_groundtruth(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: all noise repeats of one voxel, joint and per-run argmax, margins of its ground truth."""
    groundtruth_per_run = state["groundtruth_bold"][v_i]                     # (n_runs, n_tr)
    n_runs, n_tr = groundtruth_per_run.shape
    n_repeats, template_per_run_norm = state["n_repeats"], state["template_per_run_norm"]

    measurements = np.empty((n_repeats, n_runs * n_tr), dtype=np.float32)
    for rep in range(n_repeats):
        measurements[rep, :] = draw_recovery_measurement(
            groundtruth_per_run, state["noise"], state["tr_s"], v_i, rep, state["jitter"])

    # per run on its own first: best_match_indices normalises its input in place, so pass copies
    per_run_idx = np.empty((n_repeats, n_runs), dtype=np.int64)
    for r in range(n_runs):
        per_run_idx[:, r] = best_match_indices(
            template_per_run_norm[r], measurements[:, r * n_tr:(r + 1) * n_tr].copy())
    best_idx = best_match_indices(state["template_matrix_norm"], measurements)      # joint

    clean_concatenated = (groundtruth_per_run
                          - groundtruth_per_run.mean(axis=1, keepdims=True)).ravel()
    result = clean_margins(clean_concatenated, state["template_matrix_norm"], state["candidate_params"])
    result["recovered"] = state["candidate_params"][best_idx]                       # (n_repeats, 4)
    result["recovered_per_run"] = state["candidate_params"][per_run_idx]            # (n_repeats, n_runs, 4)
    return result


def run_recovery_groundtruth(
    groundtruth_bold: np.ndarray,
    candidate_params: np.ndarray,
    template_matrix_norm: np.ndarray,
    template_per_run_norm: list,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    tr_s: float,
    cf_list_hz: np.ndarray,
    n_workers: int = 1,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
) -> dict:
    """Monte Carlo recovery from the averaged ground truth, one noise level: joint fit and random effects.

    Parameters
    ----------
    groundtruth_bold : np.ndarray, shape (n_voxels, n_runs, n_tr)
        Averaged ground truth (``load_groundtruth``).
    template_matrix_norm : np.ndarray, shape (n_candidates, n_runs * n_tr)
        ``concatenate_templates`` output (joint fit).
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        ``normalize_run_templates`` output (one fit per run).
    true_params : np.ndarray, shape (n_voxels, 4)

    Returns
    -------
    dict with the keys of ``sweep_recovery_multirun.run_recovery_multirun``: recovered_params
    (n_voxels, n_repeats, 4; joint), best_idx, r_best, n_ties, margins, top_idx, top_r,
    recovered_per_run (n_voxels, n_repeats, n_runs, 4), recovered_random_effects
    (n_voxels, n_repeats, 4), recovered_first_run (n_voxels, n_repeats, 4).
    """
    n_voxels, n_runs, n_tr = groundtruth_bold.shape
    assert n_voxels == len(true_params), "one ground-truth signal per voxel"
    assert n_runs == len(template_per_run_norm), "one template matrix per run"
    assert template_per_run_norm[0].shape[1] == n_tr, "ground truth and templates differ in n_tr"
    assert template_matrix_norm.shape[1] == n_runs * n_tr
    state = dict(groundtruth_bold=groundtruth_bold, candidate_params=candidate_params,
                 template_matrix_norm=template_matrix_norm,
                 template_per_run_norm=template_per_run_norm, noise=noise, n_repeats=n_repeats,
                 tr_s=tr_s, jitter=jitter)
    if n_workers > 1:
        voxel_results = list(map_voxels(_recover_one_voxel_groundtruth, true_params, state, n_workers))
    else:
        voxel_results = [_recover_one_voxel_groundtruth(v_i, row, state)
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


# ── Report and command line ───────────────────────────────────────────────────
def recover_and_report_groundtruth(
    groundtruth_dir: Path,
    grid_dir: Path,
    output_dir: Path,
    noise_levels: list,
    n_repeats: int,
    design_seeds: Sequence[int],
    n_workers: int = 1,
) -> Path:
    """Recover from the averaged ground truth of every noise level; write all tables, plots and results.

    Each noise level is recovered against its own averaged ground truth with noise of that level.
    The "1-run" column is run 1 alone from the same data (the earlier single-run results used no
    jitter, so they are not a baseline here).

    Returns
    -------
    Path of the result folder ``<output_dir>/<timestamp>_groundtruth_<noises>_<n>runs``.
    """
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = (Path(output_dir) /
               f"{timestamp}_groundtruth_{'-'.join(noise_levels)}_{len(design_seeds)}runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_params, template_per_run = load_run_grids(Path(grid_dir), design_seeds)
    template_matrix_norm = concatenate_templates(template_per_run)
    template_per_run_norm = normalize_run_templates(template_per_run)
    logger.info("noise=%s repeats=%d runs=%d workers=%d -> %s", noise_levels, n_repeats,
                len(design_seeds), n_workers, run_dir)

    summary_by_method = {"1run": {}, "24run_joint": {}, "24run_random_effects": {}}
    recovered_by_method = {"1run": {}, "joint": {}, "random_effects": {}}   # for the tau heatmaps
    estimate_summary = {estimate: {} for estimate in ESTIMATES}
    accuracy = {estimate: {} for estimate in ESTIMATES}
    spread_by_noise = {}
    true_params = None
    for noise in noise_levels:
        groundtruth = load_groundtruth(groundtruth_dir, noise)
        assert groundtruth["design_seeds"] == list(design_seeds), (
            f"ground truth built for run orders {groundtruth['design_seeds']}, not {list(design_seeds)}")
        cf_list_hz, level_true_params = groundtruth["cf_list_hz"], groundtruth["true_params"]
        expected = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(cf_list_hz),
                                        candidate_params, len(level_true_params))
        assert np.array_equal(level_true_params, expected), (
            f"the voxels of the '{noise}' ground truth are not the MC_SEED on-grid draw")
        assert true_params is None or np.array_equal(true_params, level_true_params), \
            "noise levels were built for different voxels"
        true_params = level_true_params
        n_voxels = len(true_params)

        logger.info("noise '%s': %d voxels x %d repeats x %d runs ...", noise, n_voxels, n_repeats,
                    len(design_seeds))
        result = run_recovery_groundtruth(
            groundtruth["groundtruth_bold"], candidate_params, template_matrix_norm,
            template_per_run_norm, true_params, noise, n_repeats, groundtruth["tr_s"], cf_list_hz,
            n_workers, groundtruth["jitter"])

        recovered = {"joint": result["recovered_params"],
                     "random_effects": result["recovered_random_effects"]}
        first_run_result = {"recovered_params": result["recovered_first_run"],
                            "margins": result["margins"], "n_ties": result["n_ties"]}
        recovered_by_method["1run"][noise] = result["recovered_first_run"]
        summaries = {"1run": summarize(true_params, first_run_result, candidate_params, cf_list_hz)}
        for estimate in ESTIMATES:     # the random-effects result reuses the joint margins and ties
            recovered_by_method[estimate][noise] = recovered[estimate]
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
            saved = {k: v for k, v in result.items() if k != "recovered_params"} if estimate == "joint" \
                else {"margins": result["margins"], "n_ties": result["n_ties"]}
            np.savez(run_dir / f"sweep_groundtruth_recovery_{estimate}_results_{noise}_{timestamp}.npz",
                     true_params=true_params, recovered_params=recovered[estimate],
                     param_names=PARAM_NAMES, **saved)
            plot_recovery(true_params, recovered[estimate],
                          run_dir / f"recovery_scatter_{estimate}_{noise}.png", param_names=PARAM_NAMES)
            save_recovery_table(true_params, recovered[estimate],
                                run_dir / f"recovery_table_{estimate}_{noise}.csv", param_names=PARAM_NAMES)
            save_summary_table(summaries[f"24run_{estimate}"],
                               run_dir / f"summary_table_{estimate}_{noise}.csv")
        save_margin_table(true_params, result, run_dir / f"margin_table_{noise}.csv")

    for method, recovered_by_noise in recovered_by_method.items():      # true-vs-recovered tau heatmaps
        plot_recovered_level_distribution(
            true_params, recovered_by_noise, candidate_params,
            run_dir / f"recovered_tau_distribution_{method}.png", param_name="tau_ms",
            n_repeats=n_repeats)
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
    plot_spread_by_true_value(spread_by_noise, true_params, chance_spread(candidate_params, cf_list_hz),
                              candidate_params, run_dir / "estimate_spread_by_tau.png",
                              param_name="tau_ms", n_repeats=n_repeats)
    logger.info("done -> %s", run_dir.resolve())
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groundtruth-dir", required=True,
                        help="folder written by sweep_groundtruth_distribution.py (groundtruth_mean_<noise>.npz)")
    parser.add_argument("--grid-dir", default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise", nargs="+", default=["low", "mid", "high"],
                        choices=["low", "mid", "high"])
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    parser.add_argument("--n-workers", type=int, default=n_workers_default)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    recover_and_report_groundtruth(
        Path(args.groundtruth_dir), Path(args.grid_dir), Path(args.output_dir),
        list(dict.fromkeys(args.noise)), args.n_repeats, args.design_seeds, args.n_workers)


if __name__ == "__main__":
    main()
