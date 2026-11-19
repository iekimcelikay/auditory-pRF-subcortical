"""parallel_recovery.py
======================
Shared helpers for voxel-parallel, memory-bounded Monte Carlo recovery.

The serial ``run_recovery`` loops (one per recovery script) spend most of
their time in two places that a ``Pool`` over the candidate grid does not
help with: drawing ``N_NOISE_REPEATS`` ``PmNoise`` realisations per voxel,
and scoring them against every candidate template. Voxels are independent
(noise seeds come from ``spawn_noise_seeds(MC_SEED, v_i, rep, ...)``, not a
shared RNG stream), so voxel-level parallelism reproduces the serial result.

Memory: scoring all repeats at once builds an
``(n_candidates, n_repeats)`` float32 matrix (~19 GB for the combined
483,840-candidate grid at 10,000 repeats), so ``best_match_indices``
scores ``REPEAT_CHUNK`` repeats at a time. Per-worker peak is
``n_candidates * REPEAT_CHUNK * 4`` bytes.

Workers are forked, so the (large, read-only) template matrix is inherited
copy-on-write instead of pickled.
"""

import multiprocessing as mp
from typing import Callable, Iterator

import numpy as np

REPEAT_CHUNK = 250   # repeats scored per matmul; lower = less memory per worker

_state: dict = {}


def best_match_indices(
    template_matrix_norm: np.ndarray,
    measurements: np.ndarray,
    repeat_chunk: int = REPEAT_CHUNK,
) -> np.ndarray:
    """Index of the highest-Pearson-r candidate for each noisy repeat.

    Parameters
    ----------
    template_matrix_norm : np.ndarray, shape (n_candidates, n_timepoints)
        Mean-removed, unit-norm candidate templates.
    measurements : np.ndarray, shape (n_repeats, n_timepoints)
        Noisy measurements. Mean-removed and unit-normalised IN PLACE.
    repeat_chunk : int
        Repeats scored per matmul (bounds the score-matrix memory).

    Returns
    -------
    np.ndarray, shape (n_repeats,), dtype int64
    """
    assert template_matrix_norm.shape[1] == measurements.shape[1], (
        f"timepoint mismatch: templates {template_matrix_norm.shape[1]} "
        f"vs measurements {measurements.shape[1]}"
    )
    measurements -= measurements.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(measurements, axis=1, keepdims=True)
    measurements /= np.where(norms == 0, 1.0, norms)

    n_repeats = measurements.shape[0]
    best_idx = np.empty(n_repeats, dtype=np.int64)
    for start in range(0, n_repeats, repeat_chunk):
        stop = min(start + repeat_chunk, n_repeats)
        scores = template_matrix_norm @ measurements[start:stop].T   # (n_candidates, chunk)
        best_idx[start:stop] = np.argmax(scores, axis=0)
    return best_idx


def unit_per_run(measurements_per_run: np.ndarray) -> np.ndarray:
    """Mean-removed, unit-norm measurement of every run (each run on its own).

    Parameters
    ----------
    measurements_per_run : np.ndarray, shape (n_repeats, n_runs, n_tr)
        Not modified.

    Returns
    -------
    np.ndarray float32, shape (n_repeats, n_runs, n_tr)
        Every run has mean 0 and unit norm; an all-zero run stays zero (its r is 0, not NaN).
    """
    assert measurements_per_run.ndim == 3, f"expected (n_repeats, n_runs, n_tr), got {measurements_per_run.shape}"
    centered = measurements_per_run - measurements_per_run.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    return (centered / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def fixed_effects_scores(template_per_run_norm: list, unit_measurements: np.ndarray) -> tuple:
    """Fixed-effects score of every candidate: Pearson r of each run on its own, averaged over runs.

    The 24 per-run r tables are accumulated one run at a time, so the memory stays at two
    ``(n_candidates, n_repeats)`` matrices instead of ``n_runs`` of them.

    Parameters
    ----------
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        Mean-removed, unit-norm templates of every run (``normalize_run_templates`` output).
    unit_measurements : np.ndarray, shape (n_repeats, n_runs, n_tr)
        ``unit_per_run`` output.

    Returns
    -------
    mean_r2 : np.ndarray float32, shape (n_candidates, n_repeats)
        Mean over runs of r^2 (the score the winner is picked from).
    mean_r : np.ndarray float32, shape (n_candidates, n_repeats)
        Mean over runs of the signed r (kept to see whether a winner is anti-correlated).
    """
    n_runs = len(template_per_run_norm)
    assert unit_measurements.shape[1] == n_runs, (
        f"{unit_measurements.shape[1]} runs of measurements vs {n_runs} template matrices")
    assert all(t.shape[1] == unit_measurements.shape[2] for t in template_per_run_norm), "timepoints differ"
    n_candidates = template_per_run_norm[0].shape[0]
    mean_r2 = np.zeros((n_candidates, unit_measurements.shape[0]), dtype=np.float32)
    mean_r = np.zeros_like(mean_r2)
    for run, templates in enumerate(template_per_run_norm):
        r_run = templates @ unit_measurements[:, run, :].T            # (n_candidates, n_repeats)
        mean_r2 += r_run ** 2
        mean_r += r_run
    return mean_r2 / n_runs, mean_r / n_runs


def best_match_fixed_effects(
    template_per_run_norm: list,
    measurements_per_run: np.ndarray,
    repeat_chunk: int = REPEAT_CHUNK,
) -> dict:
    """Fixed-effects winner for each noisy repeat: the candidate with the highest mean r^2 over runs.

    Parameters
    ----------
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
    measurements_per_run : np.ndarray, shape (n_repeats, n_runs, n_tr)
        Noisy measurements; not modified.
    repeat_chunk : int
        Repeats scored per matmul (bounds the score-matrix memory).

    Returns
    -------
    dict with
        best_idx : (n_repeats,) int64, lowest index on a tie
        winner_mean_r : (n_repeats,) float32, the winner's mean SIGNED r (negative = anti-correlated)
    """
    n_repeats = measurements_per_run.shape[0]
    best_idx = np.empty(n_repeats, dtype=np.int64)
    winner_mean_r = np.empty(n_repeats, dtype=np.float32)
    for start in range(0, n_repeats, repeat_chunk):
        stop = min(start + repeat_chunk, n_repeats)
        mean_r2, mean_r = fixed_effects_scores(
            template_per_run_norm, unit_per_run(measurements_per_run[start:stop]))
        winners = np.argmax(mean_r2, axis=0)
        best_idx[start:stop] = winners
        winner_mean_r[start:stop] = mean_r[winners, np.arange(stop - start)]
    return {"best_idx": best_idx, "winner_mean_r": winner_mean_r}


def _init_worker(state: dict) -> None:
    _state.update(state)


def _voxel_task(task: tuple):
    v_i, true_row = task
    return _state["voxel_fn"](v_i, true_row, _state)


def map_voxels(
    voxel_fn: Callable,
    true_params: np.ndarray,
    state: dict,
    n_workers: int,
) -> Iterator:
    """Yield ``voxel_fn(v_i, true_row, state)`` for every voxel, in voxel order.

    Parameters
    ----------
    voxel_fn : callable
        ``(v_i, true_row, state) -> result``. Runs in a forked worker;
        ``state`` is the dict passed below (inherited, not pickled).
    true_params : np.ndarray, shape (n_voxels, n_params)
    state : dict
        Read-only data shared with workers (templates, sequences, ...).
    n_workers : int
    """
    full_state = dict(state, voxel_fn=voxel_fn)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(full_state,)) as pool:
        yield from pool.imap(_voxel_task, enumerate(true_params))
