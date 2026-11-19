"""sweep_highpass_drift_test.py
===========================
Drift-removal test (paired): does the fixed drift of the noise model cause the run-specific favourite tau?

Starting from the AVERAGED ground truth of ``sweep_groundtruth_distribution.py`` (the mean of 1,000 noisy
copies of each voxel's signal; ``groundtruth_mean_<noise>.npz``), exactly as ``sweep_recovery_from_groundtruth.py``
does: a fresh noise draw is added to it and every run is matched with the single-run Pearson recovery. Here each
run is matched twice on the SAME noisy measurement: as before ("none"), and after removing the slow DCT
components (period longer than a cutoff) from BOTH the measurement and every template of that run
("hp<cutoff>"). Identical seeds, so any difference is due to the filter. Only per-run estimates are made (a
fixed-effects estimate over runs is not formed here).

The noise drift is the sum of the first ``floor(2 * T / 120 s)`` non-constant cosine columns, so a 128 s
cutoff (the SPM default) removes nearly all of it and 64 s removes all of it, also with jittered periods.

Run from the project root.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from auditory_prf.prf_pipeline.solver.grid_recovery import MC_SEED
from auditory_prf.prf_pipeline.solver.parallel_recovery import best_match_indices, map_voxels
from auditory_prf.prf_pipeline.solver.sweep_groundtruth_distribution import NOISE_JITTER_GROUNDTRUTH
from auditory_prf.prf_pipeline.solver.sweep_recovery import PARAM_NAMES, sample_ground_truths
from auditory_prf.prf_pipeline.solver.sweep_recovery_from_groundtruth import (
    draw_recovery_measurement,
    load_groundtruth,
    recovery_signal_scale,
)
from auditory_prf.prf_pipeline.solver.sweep_recovery_multirun import (
    DEFAULT_GRID_DIR,
    DEFAULT_OUTPUT_DIR,
    RUN_DESIGN_SEEDS,
    load_run_grids,
    normalize_run_templates,
)

logger = logging.getLogger(__name__)

DEFAULT_CUTOFFS_S = [128.0, 64.0]


# ── High-pass filter ──────────────────────────────────────────────────────────
def n_highpass_columns(n_tr: int, tr_s: float, cutoff_s: float) -> int:
    """Number of DCT drift columns (the constant included) with a period longer than ``cutoff_s``.

    SPM convention: ``floor(2 * n_tr * tr_s / cutoff_s) + 1``.
    """
    return int(np.floor(2.0 * n_tr * tr_s / cutoff_s)) + 1


def highpass_projection(n_tr: int, tr_s: float, cutoff_s: float) -> np.ndarray:
    """Matrix P that removes the slow DCT components of a time course of length ``n_tr``.

    ``P = I - B B^T`` with B the orthonormal DCT-II columns 0 .. K-1 (K = ``n_highpass_columns``); column 0
    is the constant, so P also removes the mean. P is symmetric and idempotent.

    Returns
    -------
    np.ndarray float64, shape (n_tr, n_tr)
    """
    n_columns = n_highpass_columns(n_tr, tr_s, cutoff_s)
    i = np.arange(n_tr)[:, None]
    k = np.arange(n_columns)[None, :]
    basis = np.sqrt(2.0 / n_tr) * np.cos(np.pi * (2 * i + 1) * k / (2 * n_tr))
    basis[:, 0] = 1.0 / np.sqrt(n_tr)
    return np.eye(n_tr) - basis @ basis.T


def highpass_templates(template_per_run: list, projection: np.ndarray) -> list:
    """High-pass every run's templates with the same projection, then unit-normalise each row.

    Parameters
    ----------
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)
    projection : np.ndarray, shape (n_tr, n_tr)

    Returns
    -------
    list of np.ndarray float32, each (n_candidates, n_tr); rows have mean 0 and unit norm (an all-zero row
    stays zero).
    """
    filtered = []
    for templates in template_per_run:
        t = templates.astype(np.float64) @ projection
        norms = np.linalg.norm(t, axis=1, keepdims=True)
        filtered.append((t / np.where(norms == 0, 1.0, norms)).astype(np.float32))
    return filtered


# ── Paired recovery ───────────────────────────────────────────────────────────
def _voxel_task(v_i: int, true_row: np.ndarray, state: dict) -> dict:
    """Worker: new-noise measurements of one voxel's averaged ground truth, matched run by run with each method."""
    groundtruth_per_run = state["groundtruth_bold"][v_i]                    # (n_runs, n_tr), the stored average
    n_runs, n_tr = groundtruth_per_run.shape
    n_repeats = state["n_repeats"]
    measurements = np.empty((n_repeats, n_runs * n_tr), dtype=np.float32)
    for rep in range(n_repeats):
        measurements[rep] = draw_recovery_measurement(
            groundtruth_per_run, state["noise"], state["tr_s"], v_i, rep, state["jitter"],
            signal_scale=state["signal_scale"])
    out = {}
    for name, (projection, templates_per_run) in state["methods"].items():
        winners = np.empty((n_repeats, n_runs), dtype=np.int64)
        for r in range(n_runs):
            segment = measurements[:, r * n_tr:(r + 1) * n_tr]
            segment = segment @ projection if projection is not None else segment.copy()
            winners[:, r] = best_match_indices(templates_per_run[r], np.ascontiguousarray(segment))
        out[name] = state["candidate_params"][winners].astype(np.float32)     # (n_repeats, n_runs, 4)
    return out


def run_highpass_test(
    groundtruth_bold: np.ndarray,
    candidate_params: np.ndarray,
    template_per_run: list,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    cutoffs_s: Sequence[float],
    tr_s: float,
    n_workers: int = 1,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
    signal_scale: Optional[float] = 1.0,
) -> dict:
    """Single-run recovery from the averaged ground truth, with and without the high-pass filter, on identical noise.

    Parameters
    ----------
    groundtruth_bold : np.ndarray, shape (n_voxels, n_runs, n_tr)
        Averaged ground truth (``load_groundtruth``), in fractions of the mean for the additive noise.
    template_per_run : list of np.ndarray, each (n_candidates, n_tr)
        Un-normalised, mean-removed clean templates of every run order (``load_run_grids``).
    cutoffs_s : sequence of float
        High-pass cutoffs (seconds); one result per cutoff.
    signal_scale : float or None
        ``recovery_signal_scale(noise_model)``: 1.0 = additive noise on the already scaled ground truth.

    Returns
    -------
    dict ``{"none": ..., "hp<cutoff>": ...}`` of np.ndarray float32, shape (n_voxels, n_repeats, n_runs, 4)
    holding the (cf_index, alpha, tau_ms, w) recovered from each run alone.
    """
    n_voxels, n_runs, n_tr = groundtruth_bold.shape
    assert n_voxels == len(true_params) and n_runs == len(template_per_run) and template_per_run[0].shape[1] == n_tr
    methods = {"none": (None, normalize_run_templates(template_per_run))}
    for cutoff in cutoffs_s:
        projection = highpass_projection(n_tr, tr_s, cutoff)
        methods[f"hp{cutoff:g}"] = (projection.astype(np.float32), highpass_templates(template_per_run, projection))
    state = dict(groundtruth_bold=groundtruth_bold, tr_s=tr_s, candidate_params=candidate_params, methods=methods,
                 noise=noise, n_repeats=n_repeats, jitter=jitter, signal_scale=signal_scale)
    if n_workers > 1:
        voxel_results = list(map_voxels(_voxel_task, true_params, state, n_workers))
    else:
        voxel_results = [_voxel_task(v_i, row, state) for v_i, row in enumerate(true_params)]
    return {name: np.stack([r[name] for r in voxel_results]) for name in methods}


# ── Command line ──────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groundtruth-dir", required=True,
                        help="folder written by sweep_groundtruth_distribution.py (groundtruth_mean_<noise>.npz)")
    parser.add_argument("--grid-dir", default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=list(DEFAULT_CUTOFFS_S),
                        help="high-pass cutoffs in seconds (a result per cutoff, next to the unfiltered one)")
    parser.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    parser.add_argument("--n-workers", type=int, default=n_workers_default)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    groundtruth = load_groundtruth(Path(args.groundtruth_dir), args.noise)
    assert groundtruth["design_seeds"] == list(args.design_seeds), (
        f"ground truth built for run orders {groundtruth['design_seeds']}, not {list(args.design_seeds)}")
    candidate_params, template_per_run = load_run_grids(Path(args.grid_dir), args.design_seeds)
    true_params = groundtruth["true_params"]
    expected = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(groundtruth["cf_list_hz"]),
                                    candidate_params, len(true_params))
    assert np.array_equal(true_params, expected), "the ground-truth voxels are not the MC_SEED on-grid draw"
    run_dir = Path(args.output_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_highpass_test_averaged_gt_{args.noise}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("averaged ground truth of '%s' (%s model) voxels=%d repeats=%d runs=%d cutoffs=%s jitter=%s "
                "workers=%d -> %s", args.noise, groundtruth["noise_model"], len(true_params), args.n_repeats,
                len(args.design_seeds), args.cutoffs, groundtruth["jitter"], args.n_workers, run_dir)
    result = run_highpass_test(
        groundtruth["groundtruth_bold"], candidate_params, template_per_run, true_params, args.noise,
        args.n_repeats, args.cutoffs, groundtruth["tr_s"], args.n_workers, groundtruth["jitter"],
        signal_scale=recovery_signal_scale(groundtruth["noise_model"]))
    np.savez(run_dir / f"highpass_test_{args.noise}.npz", true_params=true_params,
             cutoffs_s=np.asarray(args.cutoffs), noise=args.noise, n_repeats=args.n_repeats,
             jitter=np.asarray(groundtruth["jitter"]), noise_model=groundtruth["noise_model"],
             groundtruth="averaged", param_names=np.asarray(PARAM_NAMES),
             **{f"recovered_{k}": v for k, v in result.items()})
    logger.info("done -> %s", run_dir.resolve())


if __name__ == "__main__":
    main()
