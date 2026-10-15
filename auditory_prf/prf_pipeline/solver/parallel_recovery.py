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
