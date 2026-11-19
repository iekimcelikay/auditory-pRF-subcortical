"""match_details.py
=================
What a matching keeps besides the winner: the best candidates and, for every level of every parameter, the best
score among the candidates of that level (e.g. the best fit for every tau).

Two matchers share the post-processing (``details_from_scores``):

* ``match_details``: one template matrix, Pearson r (per-run estimates, 1-run baseline).
* ``match_details_fixed_effects``: all runs; the score is the mean over runs of r^2 (each run scored on its own).

Moved here from ``sweep_noise_scale_recovery.py`` so that every sweep script can use it.
"""

import numpy as np

from auditory_prf.prf_pipeline.solver.parallel_recovery import (
    REPEAT_CHUNK,
    fixed_effects_scores,
    unit_per_run,
)

TOP_N_MATCHES = 10                              # best candidates kept for every matching


def level_indices_from_params(candidate_params: np.ndarray) -> list:
    """For every parameter, the level number (0 ..) of each candidate.

    Returns
    -------
    list of 4 np.ndarray int64, each (n_candidates,); ``np.unique`` order, so level j of a parameter is its j-th
    smallest value.
    """
    return [np.unique(candidate_params[:, p], return_inverse=True)[1].astype(np.int64).ravel()
            for p in range(candidate_params.shape[1])]


def _level_groups(level_indices: list) -> list:
    """Per parameter: (candidates sorted by level, start position of every level in that order)."""
    groups = []
    for level_index in level_indices:
        order = np.argsort(level_index, kind="stable")
        groups.append((order, np.flatnonzero(np.r_[True, np.diff(level_index[order]) != 0])))
    return groups


def details_from_scores(scores: np.ndarray, level_groups: list, top_n: int) -> dict:
    """Winner, best candidates and level profiles of a score matrix.

    Parameters
    ----------
    scores : np.ndarray, shape (n_candidates, n_chunk)
        Score of every candidate for every measurement (higher = better).
    level_groups : list
        ``_level_groups`` output.
    top_n : int
        Number of best candidates kept (at most the number of candidates).

    Returns
    -------
    dict with
        best_idx : (n_chunk,) int64, the highest score (lowest index on a tie)
        top_idx : (n_chunk, top_n) int16, the best candidates, best first
        top_r : (n_chunk, top_n) float32, their scores
        level_best_r : (n_chunk, n_levels_total) float32, the highest score among the candidates of every level
            of every parameter (the parameters' levels one after the other)
    """
    n_candidates = scores.shape[0]
    top_n = min(top_n, n_candidates)
    if top_n < n_candidates:
        candidates = np.argpartition(scores, n_candidates - top_n, axis=0)[n_candidates - top_n:]
    else:
        candidates = np.broadcast_to(np.arange(n_candidates)[:, None], scores.shape)
    candidate_r = np.take_along_axis(scores, candidates, axis=0)
    order = np.lexsort((candidates, -candidate_r), axis=0)               # best score first, lowest index on a tie
    level_best_r = np.concatenate(
        [np.maximum.reduceat(scores[sort_order], starts, axis=0).T for sort_order, starts in level_groups],
        axis=1).astype(np.float32)
    return {"best_idx": np.argmax(scores, axis=0).astype(np.int64),
            "top_idx": np.take_along_axis(candidates, order, axis=0).T.astype(np.int16),
            "top_r": np.take_along_axis(candidate_r, order, axis=0).T.astype(np.float32),
            "level_best_r": level_best_r}


def _concatenate_chunks(chunks: list) -> dict:
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


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
    centered = measurements - measurements.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.where(norms == 0, 1.0, norms)
    groups = _level_groups(level_indices)
    n_repeats = measurements.shape[0]
    chunks = []
    for start in range(0, n_repeats, repeat_chunk):
        scores = template_matrix_norm @ unit[start:min(start + repeat_chunk, n_repeats)].T   # (n_candidates, chunk)
        chunks.append(details_from_scores(scores, groups, top_n))
    return _concatenate_chunks(chunks)


def match_details_fixed_effects(
    template_per_run_norm: list,
    measurements_per_run: np.ndarray,
    level_indices: list,
    top_n: int = TOP_N_MATCHES,
    repeat_chunk: int = REPEAT_CHUNK,
    full_score_repeats: int = 0,
) -> dict:
    """Fixed-effects matching with all details: the score is the mean over runs of r^2.

    Parameters
    ----------
    template_per_run_norm : list of np.ndarray, each (n_candidates, n_tr)
        Mean-removed, unit-norm templates of every run.
    measurements_per_run : np.ndarray, shape (n_repeats, n_runs, n_tr)
        Not modified.
    level_indices : list of np.ndarray int, each (n_candidates,)
        ``level_indices_from_params``.
    full_score_repeats : int
        The mean-r^2 vector over ALL candidates is returned for the first this many repeats (capped at the number
        of repeats; 0 = none).

    Returns
    -------
    dict with the keys of ``match_details`` (``top_r`` and ``level_best_r`` hold the mean r^2) plus
        winner_mean_r : (n_repeats,) float32, the winner's mean SIGNED r (negative = anti-correlated)
        full_scores : (min(full_score_repeats, n_repeats), n_candidates) float32
    """
    n_candidates = template_per_run_norm[0].shape[0]
    assert n_candidates < np.iinfo(np.int16).max, "candidate numbers do not fit int16"
    groups = _level_groups(level_indices)
    n_repeats = measurements_per_run.shape[0]
    n_full = min(max(full_score_repeats, 0), n_repeats)
    chunks, full_scores = [], np.empty((n_full, n_candidates), dtype=np.float32)
    for start in range(0, n_repeats, repeat_chunk):
        stop = min(start + repeat_chunk, n_repeats)
        mean_r2, mean_r = fixed_effects_scores(template_per_run_norm, unit_per_run(measurements_per_run[start:stop]))
        details = details_from_scores(mean_r2, groups, top_n)
        details["winner_mean_r"] = mean_r[details["best_idx"], np.arange(stop - start)]
        chunks.append(details)
        if start < n_full:
            keep = min(stop, n_full) - start
            full_scores[start:start + keep] = mean_r2[:, :keep].T
    result = _concatenate_chunks(chunks)
    result["full_scores"] = full_scores
    return result
