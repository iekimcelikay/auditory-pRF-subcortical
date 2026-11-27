"""sweep_recovery_from_groundtruth.py
=================================
Script 2 of the averaged-noisy ground-truth experiment (design:
auditory-prf-docs-all/groundtruth_average_recovery_design.md).

Recovery of (cf_index, alpha, tau_ms, w) from the AVERAGED ground-truth signal of
``sweep_groundtruth_distribution.py`` plus fresh noise (same PmNoise and jitter, separate seed
stream), from 24 runs with a FIXED-EFFECTS estimate (every run scored on its own, the r^2 tables
averaged over runs, the highest mean r^2 wins), and the spread of the 24 per-run estimates. The best
score of every level of every parameter, the top candidates and the per-run profiles are saved, and the
full score vector over all candidates for the first few repeats. Each noise level is recovered
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

from auditory_prf.prf_pipeline.solver.additive_noise import NOISE_MODELS
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import MC_SEED, plot_recovery, save_recovery_table
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
from auditory_prf.prf_pipeline.solver.sweep_groundtruth_distribution import NOISE_JITTER_GROUNDTRUTH
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    PARAM_NAMES,
    plot_recovered_candidate_heatmap,
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
    ESTIMATES,
    RUN_DESIGN_SEEDS,
    chance_spread,
    clean_margins_fixed_effects,
    estimate_spread,
    load_run_grids,
    normalize_run_templates,
    plot_estimate_spread,
    plot_run_comparison,
    plot_spread_by_true_value,
    save_run_comparison_table,
    save_spread_table,
)
from stimuli import sweep_design_names as design_names

logger = logging.getLogger(__name__)

SEED_TAG_RECOVERY = 2                           # seed stream of the recovery noise (ground truth uses tag 1)
N_REPEATS = 1000                                 # Monte Carlo repeats per noise level


# ── Ground truth and measurement ──────────────────────────────────────────────
def load_groundtruth(groundtruth_dir: Path, noise: str) -> dict:
    """Averaged ground truth of one noise level (reads only ``groundtruth_mean_<noise>.npz``).

    Returns
    -------
    dict with ``groundtruth_bold`` (n_voxels, n_runs, n_tr) float32, ``true_params`` (n_voxels, 4),
    ``cf_list_hz`` (n_cf,), ``jitter`` (list of 3 floats), ``n_repeats`` (int), ``tr_s`` (float),
    ``design_seeds`` (list of int), ``noise_model`` ('additive' | 'legacy'; files from before the additive
    noise were built with the legacy noise).
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
                "noise_model": str(saved["noise_model"]) if "noise_model" in saved.files else "legacy",
                "n_repeats": int(saved["n_repeats"]), "tr_s": float(saved["tr_s"]),
                "design_seeds": [int(x) for x in saved["design_seeds"]]}


def recovery_signal_scale(noise_model: str) -> Optional[float]:
    """Signal scale for the recovery noise: the stored ground truth is already in fractions of the mean when
    it was built with the additive noise, so the noise is added without rescaling it (1.0); legacy: None."""
    assert noise_model in NOISE_MODELS, noise_model
    return 1.0 if noise_model == "additive" else None


def draw_recovery_measurement(
    groundtruth_per_run: np.ndarray,
    noise: str,
    tr_s: float,
    voxel_index: int,
    repeat_index: int,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
    signal_scale: Optional[float] = None,
) -> np.ndarray:
    """One noisy experiment from the averaged ground truth: independent noise per run, each run
    mean-removed, concatenated.

    Run r gets the seed ``spawn_noise_seeds(MC_SEED, SEED_TAG_RECOVERY, voxel, repeat, n_runs)[r]``.

    Parameters
    ----------
    groundtruth_per_run : np.ndarray, shape (n_runs, n_tr)
    noise : str
        'low' | 'mid' | 'high', or 'none' for the ground truth without new noise.
    signal_scale : float or None
        ``recovery_signal_scale(noise_model)``: 1.0 = additive noise on a ground truth that is already in
        fractions of the mean; None = legacy per-voxel noise scaling.

    Returns
    -------
    np.ndarray, shape (n_runs * n_tr,)
    """
    n_runs = len(groundtruth_per_run)
    run_seeds = spawn_noise_seeds(MC_SEED, SEED_TAG_RECOVERY, voxel_index, repeat_index, n_runs=n_runs)
    noisy_per_run = draw_noisy_measurement(
        list(groundtruth_per_run), noise, tr_s, run_seeds, jitter=jitter,
        signal_scale=signal_scale).reshape(n_runs, -1)
    return (noisy_per_run - noisy_per_run.mean(axis=1, keepdims=True)).ravel()


# ── Recovery ──────────────────────────────────────────────────────────────────
def _recover_one_voxel_groundtruth(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: all noise repeats of one voxel, fixed-effects and per-run matching, margins of its ground truth."""
    groundtruth_per_run = state["groundtruth_bold"][v_i]                     # (n_runs, n_tr)
    n_runs, n_tr = groundtruth_per_run.shape
    n_repeats, template_per_run_norm = state["n_repeats"], state["template_per_run_norm"]

    measurements = np.empty((n_repeats, n_runs * n_tr), dtype=np.float32)
    for rep in range(n_repeats):
        measurements[rep, :] = draw_recovery_measurement(
            groundtruth_per_run, state["noise"], state["tr_s"], v_i, rep, state["jitter"],
            signal_scale=state.get("signal_scale"))
    measurements = measurements.reshape(n_repeats, n_runs, n_tr)
    clean_per_run = groundtruth_per_run                                              # for the margins
    projection = state.get("projection")
    if projection is not None:     # same noise draws, slow components of every run removed (templates filtered too)
        measurements = measurements @ projection.astype(np.float32)
        clean_per_run = groundtruth_per_run @ projection

    level_indices, top_n = state["level_indices"], TOP_N_MATCHES
    fixed = match_details_fixed_effects(template_per_run_norm, measurements, level_indices, top_n,
                                        full_score_repeats=state["full_score_repeats"])
    per_run = [match_details(template_per_run_norm[r], measurements[:, r, :], level_indices, top_n)
               for r in range(n_runs)]                    # each run on its own: the single-run estimate (argmax r)

    result = clean_margins_fixed_effects(clean_per_run, template_per_run_norm, state["candidate_params"])
    result["recovered"] = state["candidate_params"][fixed["best_idx"]]              # (n_repeats, 4)
    result["winner_mean_r"] = fixed["winner_mean_r"]                                # (n_repeats,)
    result["recovered_per_run"] = state["candidate_params"][
        np.stack([d["best_idx"] for d in per_run], axis=1)]                         # (n_repeats, n_runs, 4)
    for name in ("top_idx", "top_r", "level_best_r"):     # every match: fixed effects and run by run
        result[f"match_{name}_fixed_effects"] = fixed[name]
        result[f"match_{name}_per_run"] = np.stack([d[name] for d in per_run], axis=1)
    if state["full_score_repeats"] > 0:
        result["full_scores_fixed_effects"] = fixed["full_scores"]                  # (n_full, n_candidates)
    return result


def run_recovery_groundtruth(
    groundtruth_bold: np.ndarray,
    candidate_params: np.ndarray,
    template_per_run_norm: list,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    tr_s: float,
    n_workers: int = 1,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
    signal_scale: Optional[float] = None,
    full_score_repeats: int = 0,
    projection: Optional[np.ndarray] = None,
) -> dict:
    """Monte Carlo recovery from the averaged ground truth, one noise level: the fixed-effects estimate.

    Parameters
    ----------
    groundtruth_bold : np.ndarray, shape (n_voxels, n_runs, n_tr)
        Averaged ground truth (``load_groundtruth``).
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        ``normalize_run_templates`` output (one fit per run).
    true_params : np.ndarray, shape (n_voxels, 4)
    full_score_repeats : int
        The mean-r^2 vector over all candidates is kept for the first this many repeats of every voxel (0 = none).
    projection : np.ndarray, shape (n_tr, n_tr), optional
        High-pass projection (``highpass.highpass_projection``) applied to every run of the noisy measurement and
        of the clean ground truth used for the margins. ``template_per_run_norm`` must then be made from templates
        filtered with the same projection (``highpass.highpass_template_per_run``). None = no filter.

    Returns
    -------
    dict with the keys of ``sweep_recovery_multirun.run_recovery_multirun``: recovered_params
    (n_voxels, n_repeats, 4; fixed effects), winner_mean_r (n_voxels, n_repeats), best_idx, r_best, n_ties,
    margins, top_idx, top_r (the clean ground truth on the mean-r^2 score), recovered_per_run
    (n_voxels, n_repeats, n_runs, 4), recovered_first_run (n_voxels, n_repeats, 4),
    match_{top_idx,top_r,level_best_r}_{fixed_effects,per_run}, and full_scores_fixed_effects
    (n_voxels, min(full_score_repeats, n_repeats), n_candidates) if ``full_score_repeats > 0``.
    """
    n_voxels, n_runs, n_tr = groundtruth_bold.shape
    assert n_voxels == len(true_params), "one ground-truth signal per voxel"
    assert n_runs == len(template_per_run_norm), "one template matrix per run"
    assert template_per_run_norm[0].shape[1] == n_tr, "ground truth and templates differ in n_tr"
    state = dict(groundtruth_bold=groundtruth_bold, candidate_params=candidate_params,
                 template_per_run_norm=template_per_run_norm, noise=noise, n_repeats=n_repeats,
                 tr_s=tr_s, jitter=jitter, signal_scale=signal_scale, full_score_repeats=full_score_repeats,
                 projection=projection, level_indices=level_indices_from_params(candidate_params))
    if n_workers > 1:
        voxel_results = list(map_voxels(_recover_one_voxel_groundtruth, true_params, state, n_workers))
    else:
        voxel_results = [_recover_one_voxel_groundtruth(v_i, row, state)
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


# ── Report and command line ───────────────────────────────────────────────────
def recover_and_report_groundtruth(
    groundtruth_dir: Path,
    grid_dir: Path,
    output_dir: Path,
    noise_levels: list,
    n_repeats: int,
    design_seeds: Sequence[int],
    n_workers: int = 1,
    full_score_repeats: int = 5,
    highpass_cutoff_s: Optional[float] = None,
    design_name: Optional[str] = None,
    design_label: Optional[str] = None,
) -> Path:
    """Recover from the averaged ground truth of every noise level; write all tables, plots and results.

    ``design_name`` / ``design_label`` (stimuli/sweep_design_names.py) are stored in every result file and the label is
    shown in the title of every plot; None = neither.

    ``full_score_repeats``: the full mean-r^2 vector over all candidates is saved for the first this many repeats
    of every voxel (0 = none).

    Each noise level is recovered against its own averaged ground truth with noise of that level.
    The "1-run" column is run 1 alone from the same data (the earlier single-run results used no
    jitter, so they are not a baseline here).

    ``highpass_cutoff_s``: None = the unfiltered recovery (unchanged). A number = the same recovery after
    removing the slow DCT components (period longer than the cutoff, seconds) from the measurement, the clean
    ground truth and the templates of every run, on the same noise seeds; the result folder gets the suffix
    ``_hp<cutoff>`` and the npz files the key ``highpass_cutoff_s``.

    Returns
    -------
    Path of the result folder ``<output_dir>/<timestamp>_groundtruth_<noises>_<n>runs`` (``..._hp<cutoff>`` for a
    filtered pass).
    """
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    filter_suffix = "" if highpass_cutoff_s is None else f"_{filter_name(highpass_cutoff_s)}"
    run_dir = (Path(output_dir) /
               f"{timestamp}_groundtruth_{'-'.join(noise_levels)}_{len(design_seeds)}runs{filter_suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_params, template_per_run = load_run_grids(Path(grid_dir), design_seeds)
    # a filtered pass builds its templates per noise level, once the sampling interval of the ground truth is known
    template_per_run_norm = normalize_run_templates(template_per_run) if highpass_cutoff_s is None else None
    filter_saved = {} if highpass_cutoff_s is None else {"highpass_cutoff_s": highpass_cutoff_s}
    logger.info("noise=%s repeats=%d runs=%d workers=%d -> %s", noise_levels, n_repeats,
                len(design_seeds), n_workers, run_dir)

    summary_by_method = {"1run": {}, "24run_fixed_effects": {}}
    recovered_by_method = {"1run": {}, "fixed_effects": {}}   # for the tau heatmaps
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

        logger.info("noise '%s' (%s model): %d voxels x %d repeats x %d runs ...", noise,
                    groundtruth["noise_model"], n_voxels, n_repeats, len(design_seeds))
        projection = None
        if highpass_cutoff_s is not None:
            check_cutoff(template_per_run[0].shape[1], groundtruth["tr_s"], highpass_cutoff_s)
            projection = highpass_projection(template_per_run[0].shape[1], groundtruth["tr_s"], highpass_cutoff_s)
            template_per_run_norm = normalize_run_templates(highpass_template_per_run(template_per_run, projection))
        result = run_recovery_groundtruth(
            groundtruth["groundtruth_bold"], candidate_params,
            template_per_run_norm, true_params, noise, n_repeats, groundtruth["tr_s"],
            n_workers, groundtruth["jitter"],
            signal_scale=recovery_signal_scale(groundtruth["noise_model"]),
            full_score_repeats=full_score_repeats, projection=projection)
        logger.info("[%s] fixed-effects winner has a negative mean r in %.4f of the repeats", noise,
                    float(np.mean(result["winner_mean_r"] < 0)))

        recovered = {"fixed_effects": result["recovered_params"]}
        first_run_result = {"recovered_params": result["recovered_first_run"],
                            "margins": result["margins"], "n_ties": result["n_ties"]}
        recovered_by_method["1run"][noise] = result["recovered_first_run"]
        summaries = {"1run": summarize(true_params, first_run_result, candidate_params, cf_list_hz)}
        for estimate in ESTIMATES:     # margins and ties describe the clean ground truth on the mean-r^2 score
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
            saved = {k: v for k, v in result.items() if k != "recovered_params"}
            np.savez(run_dir / f"sweep_groundtruth_recovery_{estimate}_results_{noise}_{timestamp}.npz",
                     true_params=true_params, recovered_params=recovered[estimate],
                     param_names=PARAM_NAMES, noise_model=groundtruth["noise_model"],
                     design=design_name or "", design_label=design_label or "", **filter_saved, **saved)
            plot_recovery(true_params, recovered[estimate],
                          run_dir / f"recovery_scatter_{estimate}_{noise}.png", param_names=PARAM_NAMES,
                          design_label=design_label)
            save_recovery_table(true_params, recovered[estimate],
                                run_dir / f"recovery_table_{estimate}_{noise}.csv", param_names=PARAM_NAMES)
            save_summary_table(summaries[f"24run_{estimate}"],
                               run_dir / f"summary_table_{estimate}_{noise}.csv")
        save_margin_table(true_params, result, run_dir / f"margin_table_{noise}.csv")

    for method, recovered_by_noise in recovered_by_method.items():      # true-vs-recovered tau heatmaps
        plot_recovered_level_distribution(
            true_params, recovered_by_noise, candidate_params,
            run_dir / f"recovered_tau_distribution_{method}.png", param_name="tau_ms",
            n_repeats=n_repeats, design_label=design_label)
        plot_recovered_candidate_heatmap(                  # only written when the candidate grid is finer than the ground truth
            true_params, recovered_by_noise, candidate_params,
            run_dir / f"recovered_tau_candidate_heatmap_{method}.png", param_name="tau_ms",
            n_repeats=n_repeats, design_label=design_label)
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
    parser.add_argument("--groundtruth-dir", required=True,
                        help="folder written by sweep_groundtruth_distribution.py (groundtruth_mean_<noise>.npz)")
    parser.add_argument("--design", choices=design_names.DESIGN_NAMES, default=design_names.DEFAULT_DESIGN_NAME,
                        help="Sweep design (stimuli/sweep_design_names.py): picks the default folders and the label "
                             "shown in the plots; the grids and the ground truth must be of this design.")
    parser.add_argument("--grid-dir", default=None, help="Default: sweep_recovery_multirun_grids/<design>.")
    parser.add_argument("--output-dir", default=None, help="Default: sweep_recovery_results/<design>.")
    parser.add_argument("--noise", nargs="+", default=["low", "mid", "high"],
                        choices=["low", "mid", "high"])
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    parser.add_argument("--n-workers", type=int, default=n_workers_default)
    parser.add_argument("--full-score-repeats", type=int, default=5,
                        help="repeats per voxel whose full r^2 vector over all candidates is saved (0 = none)")
    parser.add_argument("--highpass-cutoffs", type=float, nargs="*", default=list(DEFAULT_HIGHPASS_CUTOFFS_S),
                        help="high-pass cutoffs in seconds; each one is recovered in its own result folder "
                             "(suffix _hp<cutoff>) next to the unfiltered recovery, on the same noise; "
                             "give the option without values (or 0) for the unfiltered recovery only")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for cutoff_s in [None, *parse_cutoffs(args.highpass_cutoffs)]:         # unfiltered first, then each filter
        recover_and_report_groundtruth(
            Path(args.groundtruth_dir), Path(args.grid_dir or design_names.grid_dir(args.design)),
            Path(args.output_dir or design_names.results_dir(args.design)),
            list(dict.fromkeys(args.noise)), args.n_repeats, args.design_seeds, args.n_workers,
            full_score_repeats=args.full_score_repeats, highpass_cutoff_s=cutoff_s,
            design_name=args.design, design_label=design_names.design_label(args.design))


if __name__ == "__main__":
    main()
