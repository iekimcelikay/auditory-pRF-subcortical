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

Run from the project root:
    python -m auditory_prf.prf_pipeline.solver.sweep_recovery --noise none --gt-mode on_grid
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
) -> tuple:
    """Forward-simulate every (cf_index, alpha, tau_ms, w) candidate (noiseless).

    Each (alpha, cf_index) run train is assembled once; (tau, w) only change AdapTrans
    and the HRF stage.

    Returns
    -------
    candidate_params : np.ndarray, shape (n_candidates, 4), order alpha > CF > tau > w
    template_matrix_norm : np.ndarray float32, shape (n_candidates, n_timepoints)
        Mean-removed, unit-norm templates.
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
                                                      run_seeds)
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
                 template_matrix_norm=template_matrix_norm, noise=noise, n_repeats=n_repeats)
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


def summarize(true_params: np.ndarray, result: dict) -> dict:
    """Printable summary: exact-recovery fraction and margin quantiles per parameter."""
    recovered = result["recovered_params"]
    summary = {"exact_fraction": {}, "margin_median": {}, "margin_min": {}}
    for p, name in enumerate(PARAM_NAMES):
        summary["exact_fraction"][name] = float(np.isclose(recovered[:, :, p],
                                                           true_params[:, None, p]).mean())
        summary["margin_median"][name] = float(np.nanmedian(result["margins"][:, p]))
        summary["margin_min"][name] = float(np.nanmin(result["margins"][:, p]))
    summary["n_ties_max"] = int(result["n_ties"].max())
    return summary


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
    parser.add_argument("--noise", choices=["none", "low", "mid", "high"], default="none")
    parser.add_argument("--gt-mode", choices=["on_grid", "off_grid"], default="on_grid")
    parser.add_argument("--n-voxels", type=int, default=N_VOXELS)
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Default: 1 for --noise none, else N_NOISE_REPEATS.")
    parser.add_argument("--design-seed", type=int, default=DESIGN_SEED)
    parser.add_argument("--n-workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--grid-file", type=str, default=None,
                        help="Load a saved candidate_grid_*.npz instead of rebuilding.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import stimuli.sweep_design_erb as design   # needs the project root on sys.path
    n_repeats = args.n_repeats or (1 if args.noise == "none" else N_NOISE_REPEATS)
    run_dir = (Path(args.output_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{args.noise}_"
               f"{args.gt_mode}_design{args.design_seed}")   # seed keeps array tasks apart
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("noise=%s gt_mode=%s voxels=%d repeats=%d workers=%d -> %s", args.noise,
                args.gt_mode, args.n_voxels, n_repeats, args.n_workers, run_dir)

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
    result = run_recovery(raw_psth_by_seq, run_design, total_run_dur_s, tr_s,
                          design.SILENCE_SEQ_ID, hrf_kernel, candidate_params,
                          template_matrix_norm, true_params, args.noise, n_repeats, args.n_workers)

    recovered = result["recovered_params"]
    accuracy = compute_accuracy_correlations(true_params, recovered, param_names=PARAM_NAMES)
    summary = summarize(true_params, result)
    logger.info("accuracy (true vs per-voxel mean recovered, Pearson r): %s",
                {k: round(v, 4) for k, v in accuracy.items()})
    logger.info("exact recovery fraction: %s", {k: round(v, 3) for k, v in summary["exact_fraction"].items()})
    logger.info("clean-BOLD margin median: %s", {k: f"{v:.2e}" for k, v in summary["margin_median"].items()})
    logger.info("clean-BOLD margin min   : %s", {k: f"{v:.2e}" for k, v in summary["margin_min"].items()})
    logger.info("max candidates tied with the best (r within %.0e): %d", TIE_TOL, summary["n_ties_max"])

    # the old argmax structure, then the margins
    np.savez(run_dir / f"recovery_results_design{args.design_seed}_{args.noise}.npz",
             true_params=true_params, recovered_params=recovered,
             param_names=PARAM_NAMES, **{k: v for k, v in result.items() if k != "recovered_params"})
    plot_recovery(true_params, recovered, run_dir / f"recovery_scatter_{args.noise}.png",
                  param_names=PARAM_NAMES)
    save_recovery_table(true_params, recovered, run_dir / f"recovery_table_{args.noise}.csv",
                        param_names=PARAM_NAMES)
    save_accuracy_table({args.noise: accuracy}, run_dir / f"accuracy_table_{args.noise}.csv",
                        param_names=PARAM_NAMES)
    plot_accuracy_comparison({args.noise: accuracy}, run_dir / f"accuracy_comparison_{args.noise}.png",
                             param_names=PARAM_NAMES, n_voxels=args.n_voxels, n_repeats=n_repeats)
    save_margin_table(true_params, result, run_dir / f"margin_table_{args.noise}.csv")
    logger.info("done -> %s", run_dir.resolve())


if __name__ == "__main__":
    main()
