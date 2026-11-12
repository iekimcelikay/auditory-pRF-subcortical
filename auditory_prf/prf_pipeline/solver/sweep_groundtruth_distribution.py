"""sweep_groundtruth_distribution.py
=================================
Script 1 of the averaged-noisy ground-truth experiment (design:
auditory-prf-docs-all/groundtruth_average_recovery_design.md).

For every voxel, run order and noise level: ``n_repeats`` independent noisy copies of the clean BOLD
of each of the 24 runs (``PmNoise`` with jitter). The full distribution is saved (one ``.npy`` per
noise level) and its mean over the repeats is the new ground-truth signal (one ``.npz`` per noise
level). No recovery happens here; ``sweep_recovery_from_groundtruth.py`` does that.

Run from the project root.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from auditory_prf.prf_pipeline.hrf import SUBCORTICAL_PARAMS, build_hrf_kernel
from auditory_prf.prf_pipeline.run_assembly_sweeps import (
    SIGNAL_DT_S,
    load_raw_psths,
    run_sweep_forward_model,
)
from auditory_prf.prf_pipeline.solver.forward_model import draw_noisy_measurement, spawn_noise_seeds
from auditory_prf.prf_pipeline.solver.grid_recovery import MC_SEED
from auditory_prf.prf_pipeline.solver.parallel_recovery import map_voxels
from auditory_prf.prf_pipeline.solver.sweep_recovery import (
    N_VOXELS,
    PARAM_NAMES,
    RECTIFY,
    RHO,
    sample_ground_truths,
)
from auditory_prf.prf_pipeline.solver.sweep_recovery_multirun import (
    DEFAULT_GRID_DIR,
    DEFAULT_RESULTS_DIR,
    RUN_DESIGN_SEEDS,
    build_run_designs,
)

logger = logging.getLogger(__name__)

NOISE_JITTER_GROUNDTRUTH = [0.1, 0.1, 0.1]       # PmNoise jitter (frequency, amplitude, phase), fraction of nominal
N_REPEATS = 1000                                 # noisy copies per voxel, run and noise level
NOISE_LEVELS = ("low", "mid", "high")
SEED_TAG_GROUNDTRUTH = 1                         # seed stream of these draws (recovery uses tag 2)
DEFAULT_OUTPUT_DIR = "sweep_groundtruth_results"


# ── Draws ─────────────────────────────────────────────────────────────────────
def compute_clean_bold_per_run(
    true_row: np.ndarray,
    raw_psth_by_seq: dict,
    run_designs: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
) -> np.ndarray:
    """Clean BOLD of one voxel in every run order.

    Parameters
    ----------
    true_row : np.ndarray, shape (4,)
        (cf_index, alpha, tau_ms, w).

    Returns
    -------
    np.ndarray, shape (n_runs, n_tr)
    """
    cf_index, alpha, tau_ms, w = true_row
    return np.stack([
        run_sweep_forward_model(
            raw_psth_by_seq, int(round(cf_index)), run_design, total_run_dur_s, tr_s,
            silence_seq_id, alpha=alpha, w=w, rectify=RECTIFY, rho=RHO, tau_ms=tau_ms,
            hrf_kernel=hrf_kernel)["bold_combined"]
        for run_design in run_designs])


def draw_groundtruth_repeat(
    clean_bold_per_run: np.ndarray,
    noise: str,
    tr_s: float,
    voxel_index: int,
    repeat_index: int,
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
) -> np.ndarray:
    """One noisy copy of the clean runs: independent noise per run (not mean-removed).

    Run r gets the seed ``spawn_noise_seeds(MC_SEED, SEED_TAG_GROUNDTRUTH, voxel, repeat, n_runs)[r]``.

    Parameters
    ----------
    clean_bold_per_run : np.ndarray, shape (n_runs, n_tr)
    noise : str
        'low' | 'mid' | 'high', or 'none' for the clean BOLD.

    Returns
    -------
    np.ndarray float32, shape (n_runs, n_tr)
    """
    n_runs = len(clean_bold_per_run)
    run_seeds = spawn_noise_seeds(MC_SEED, SEED_TAG_GROUNDTRUTH, voxel_index, repeat_index,
                                  n_runs=n_runs)
    noisy = draw_noisy_measurement(list(clean_bold_per_run), noise, tr_s, run_seeds, jitter=jitter)
    return noisy.reshape(n_runs, -1).astype(np.float32)


def _build_voxel_distribution(v_i: int, true_row: np.ndarray, state: dict) -> tuple:
    """Worker: all noisy copies of one voxel, and its clean BOLD.

    Returns
    -------
    distribution : np.ndarray float32, shape (n_repeats, n_runs, n_tr)
    clean_bold_per_run : np.ndarray float32, shape (n_runs, n_tr)
    """
    clean_bold_per_run = compute_clean_bold_per_run(
        true_row, state["raw_psth_by_seq"], state["run_designs"], state["total_run_dur_s"],
        state["tr_s"], state["silence_seq_id"], state["hrf_kernel"])
    n_runs, n_tr = clean_bold_per_run.shape
    distribution = np.empty((state["n_repeats"], n_runs, n_tr), dtype=np.float32)
    for repeat_index in range(state["n_repeats"]):
        distribution[repeat_index] = draw_groundtruth_repeat(
            clean_bold_per_run, state["noise"], state["tr_s"], v_i, repeat_index, state["jitter"])
    return distribution, clean_bold_per_run.astype(np.float32)


# ── Distribution and its mean ─────────────────────────────────────────────────
def load_candidate_params(grid_dir: Path, design_seed: int) -> np.ndarray:
    """Candidate parameters (cf_index, alpha, tau_ms, w) of a saved run grid.

    Only the parameters are read; the templates in the file are left on disk.

    Returns
    -------
    np.ndarray, shape (n_candidates, 4)
    """
    with np.load(Path(grid_dir) / f"run_grid_design{design_seed}.npz") as grid:
        return grid["candidate_params"]


def build_groundtruth_distribution(
    raw_psth_by_seq: dict,
    cf_list_hz: np.ndarray,
    run_designs: list,
    total_run_dur_s: float,
    tr_s: float,
    silence_seq_id: str,
    hrf_kernel: np.ndarray,
    true_params: np.ndarray,
    noise: str,
    n_repeats: int,
    output_dir: Path,
    design_seeds: Sequence[int],
    jitter: list = NOISE_JITTER_GROUNDTRUTH,
    n_workers: int = 1,
) -> dict:
    """Save the distribution of noisy signals of every voxel and its mean (the new ground truth).

    Voxels are streamed into a memory-mapped ``groundtruth_distribution_<noise>.npy`` (float32,
    (n_voxels, n_repeats, n_runs, n_tr)); ``groundtruth_mean_<noise>.npz`` is written LAST, so its
    existence means the distribution is complete. Mean and SD are over the repeats (ddof 0). The
    ``.npz`` also holds ``clean_bold``: the PmNoise drift is not zero-mean, so the mean is the clean
    signal plus a small deterministic drift, and ``groundtruth_bold - clean_bold`` shows it.

    Parameters
    ----------
    true_params : np.ndarray, shape (n_voxels, 4)
        (cf_index, alpha, tau_ms, w) of every voxel.
    noise : str
        'low' | 'mid' | 'high' (or 'none' for a noise-free check).

    Returns
    -------
    dict with ``distribution_path`` and ``mean_path`` (Path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_voxels, n_runs = len(true_params), len(run_designs)
    n_tr = int(round(total_run_dur_s / tr_s))
    state = dict(raw_psth_by_seq=raw_psth_by_seq, run_designs=run_designs,
                 total_run_dur_s=total_run_dur_s, tr_s=tr_s, silence_seq_id=silence_seq_id,
                 hrf_kernel=hrf_kernel, noise=noise, n_repeats=n_repeats, jitter=jitter)
    distribution_path = output_dir / f"groundtruth_distribution_{noise}.npy"
    mean_path = output_dir / f"groundtruth_mean_{noise}.npz"
    logger.info("[%s] %d voxels x %d repeats x %d runs x %d TRs = %.2f GB -> %s", noise, n_voxels,
                n_repeats, n_runs, n_tr, n_voxels * n_repeats * n_runs * n_tr * 4 / 1e9,
                distribution_path)

    distribution = np.lib.format.open_memmap(
        distribution_path, mode="w+", dtype=np.float32, shape=(n_voxels, n_repeats, n_runs, n_tr))
    groundtruth_bold = np.empty((n_voxels, n_runs, n_tr), dtype=np.float32)
    groundtruth_sd = np.empty_like(groundtruth_bold)
    clean_bold = np.empty_like(groundtruth_bold)
    if n_workers > 1:
        voxel_results = map_voxels(_build_voxel_distribution, true_params, state, n_workers)
    else:
        voxel_results = (_build_voxel_distribution(v_i, row, state)
                         for v_i, row in enumerate(true_params))
    for v_i, (voxel_distribution, voxel_clean_bold) in enumerate(voxel_results):
        assert voxel_distribution.shape == distribution.shape[1:], voxel_distribution.shape
        distribution[v_i] = voxel_distribution
        clean_bold[v_i] = voxel_clean_bold
        groundtruth_bold[v_i] = voxel_distribution.mean(axis=0, dtype=np.float64)
        groundtruth_sd[v_i] = voxel_distribution.std(axis=0, dtype=np.float64)
        logger.info("[%s] voxel %d/%d done", noise, v_i + 1, n_voxels)
    distribution.flush()
    del distribution

    np.savez(mean_path, groundtruth_bold=groundtruth_bold, groundtruth_sd=groundtruth_sd,
             clean_bold=clean_bold, true_params=true_params, cf_list_hz=cf_list_hz, noise=noise,
             jitter=np.asarray(jitter, dtype=float), n_repeats=n_repeats,
             design_seeds=np.asarray(design_seeds), seed_tag=SEED_TAG_GROUNDTRUTH, tr_s=tr_s,
             param_names=np.asarray(PARAM_NAMES))
    logger.info("[%s] ground truth (mean of %d draws) saved -> %s", noise, n_repeats,
                mean_path.resolve())
    return {"distribution_path": distribution_path, "mean_path": mean_path}


# ── Command line ──────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    n_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--grid-dir", default=DEFAULT_GRID_DIR,
                        help="folder with run_grid_design<seed>.npz (only the candidate parameters are read, "
                             "to draw the same on-grid voxels as the earlier results)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise", nargs="+", default=list(NOISE_LEVELS),
                        choices=["low", "mid", "high"])
    parser.add_argument("--n-voxels", type=int, default=N_VOXELS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--design-seeds", type=int, nargs="+", default=list(RUN_DESIGN_SEEDS))
    parser.add_argument("--jitter", type=float, nargs=3, default=list(NOISE_JITTER_GROUNDTRUTH),
                        metavar=("FREQ", "AMP", "PHASE"), help="PmNoise jitter, fraction of nominal")
    parser.add_argument("--n-workers", type=int, default=n_workers_default)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import stimuli.sweep_design_erb as design   # needs the project root on sys.path
    raw_psth_by_seq, cf_list_hz = load_raw_psths(Path(args.results_dir))
    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=SIGNAL_DT_S, duration=32.0)
    run_designs, total_run_dur_s = build_run_designs(args.design_seeds)
    candidate_params = load_candidate_params(Path(args.grid_dir), args.design_seeds[0])
    true_params = sample_ground_truths("on_grid", np.random.default_rng(MC_SEED), len(cf_list_hz),
                                       candidate_params, args.n_voxels)

    run_dir = (Path(args.output_dir) /
               f"{datetime.now():%Y%m%d_%H%M%S}_groundtruth_{len(args.design_seeds)}runs")
    logger.info("noise=%s voxels=%d repeats=%d runs=%d jitter=%s workers=%d -> %s", args.noise,
                args.n_voxels, args.n_repeats, len(args.design_seeds), args.jitter, args.n_workers,
                run_dir)
    for noise in dict.fromkeys(args.noise):
        build_groundtruth_distribution(
            raw_psth_by_seq, cf_list_hz, run_designs, total_run_dur_s, design.TR_S,
            design.SILENCE_SEQ_ID, hrf_kernel, true_params, noise, args.n_repeats, run_dir,
            args.design_seeds, jitter=args.jitter, n_workers=args.n_workers)
    logger.info("done -> %s", run_dir.resolve())


if __name__ == "__main__":
    main()
