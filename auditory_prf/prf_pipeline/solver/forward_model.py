"""forward_model.py
====================
Shared, plotting-free forward model for the tone-cloud AdapTrans+BOLD
pipeline, factored out of ``full_pipeline_toneclouds_adaptrans_claude.py``
so that grid-search / parameter-recovery code (which calls the forward
model tens of thousands of times) does not duplicate that script's
plotting-and-saving-entangled orchestration, nor re-implement its own copy
of the same stage sequence (as ``tau_recovery_sanity_check.py`` currently
does).

Stage order (see CLAUDE.md): load cochlear PSTH -> power-law sharpening
(alpha) -> chunk into tone-ON windows -> boxcar train -> AdapTrans ON/OFF
(tau_on_ms / tau_off_ms, adaptation weight w) -> HRF convolution -> BOLD.

Caching contract (why functions are split this way)
----------------------------------------------------
- Loading a .npz is independent of every model parameter -> load once per
  sequence (``load_raw_sequences``).
- Power-law sharpening rescales so the mean is preserved **across the
  sequence's full population PSTH** (all CFs x all time bins) -> it must be
  applied to the whole ``population_psth`` matrix, and only then indexed by
  CF row. It depends only on ``alpha`` -> compute once per (seq_id, alpha)
  (``sharpen_all_sequences``), reused across every CF index.
- Chunking + boxcar-train construction depends on (seq_id, alpha, cf_index)
  but not on tau_on/tau_off/w -> cache once per (alpha, cf_index)
  (``build_per_seq_trains``).
- AdapTrans filtering + HRF convolution is the only stage that depends on
  the full (cf_index, alpha, tau_on, tau_off, w) tuple -> this is the one
  call that actually runs once per grid candidate (``forward_bold_runs``).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold
from auditory_prf.utils.result_saver import ResultSaver
from prf_models.pm_noise import PmNoise, apply_bold_noise


def load_raw_sequences(results_dir: Union[str, Path]) -> tuple[dict, np.ndarray]:
    """Load every sequence's raw (unsharpened) population PSTH, once.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``wav*/**/*.npz`` cochlear-simulation outputs
        (one .npz per stimulus sequence, each holding all CFs).

    Returns
    -------
    raw_sequences : dict
        ``{seq_id: {"population_psth": (n_cf, n_bins), "time_axis_s": (n_bins,)}}``
    cf_list_hz : np.ndarray, shape (n_cf,)
        Characteristic frequencies (Hz) for each row of every
        ``population_psth``. Assumed identical across sequences.
    """
    results_dir = Path(results_dir)
    npz_files = sorted(results_dir.glob("wav*/**/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {results_dir}/wav*/")

    raw_sequences: dict = {}
    cf_list_hz: Optional[np.ndarray] = None

    for npz_path in npz_files:
        population_psth, time_axis_s, _cf_index, _cf_hz, seq_id = load_population_psth(npz_path, 0)
        raw_sequences[seq_id] = {
            "population_psth": population_psth,
            "time_axis_s": time_axis_s,
        }
        if cf_list_hz is None:
            data = ResultSaver(npz_path.parent).load_npz(npz_path.name)
            cf_list_hz = np.asarray(data["cf_list"])

    return raw_sequences, cf_list_hz


def sharpen_all_sequences(raw_sequences: dict, alpha: float) -> dict:
    """Power-law sharpen every sequence's population PSTH at one ``alpha``.

    Must run on the full ``(n_cf, n_bins)`` matrix per sequence (not a
    single CF row) because the mean-preserving rescale in
    ``apply_powerlaw_population`` is computed over all CFs x all time bins.

    Parameters
    ----------
    raw_sequences : dict
        Output of ``load_raw_sequences``.
    alpha : float
        Power-law sharpening exponent.

    Returns
    -------
    dict
        ``{seq_id: sharpened_population_psth (n_cf, n_bins)}``
    """
    return {
        seq_id: apply_powerlaw_population(entry["population_psth"], alpha)
        for seq_id, entry in raw_sequences.items()
    }


def build_per_seq_trains(
    raw_sequences: dict,
    sharpened_by_seq: dict,
    cf_index: int,
    cf_list_hz: np.ndarray,
    silence_seq_id: str,
    chunk_margin_ms: float = 30.0,
) -> tuple[dict, float]:
    """Build the boxcar-train ``per_seq`` dict for one (alpha, cf_index) pair.

    Parameters
    ----------
    raw_sequences : dict
        Output of ``load_raw_sequences`` (supplies ``time_axis_s``).
    sharpened_by_seq : dict
        Output of ``sharpen_all_sequences`` for the desired alpha.
    cf_index : int
        Zero-based row index into each sequence's population PSTH.
    cf_list_hz : np.ndarray, shape (n_cf,)
        Characteristic frequencies, as returned by ``load_raw_sequences``.
    silence_seq_id : str
        seq_id of the silence/null-trial stimulus, handled as a flat train
        at its mean rate rather than chunked (it has no tone-ON windows).
    chunk_margin_ms : float
        Extra window after tone offset passed to ``chunk_from_id`` (ms).

    Returns
    -------
    per_seq : dict
        ``{seq_id: {"train": np.ndarray}}``, ready for ``assemble_run_bold``.
    cf_hz : float
        Characteristic frequency (Hz) of ``cf_index``.
    """
    cf_hz = float(cf_list_hz[cf_index])
    per_seq: dict = {}

    for seq_id, sharpened_population in sharpened_by_seq.items():
        time_axis_s = raw_sequences[seq_id]["time_axis_s"]
        sharpened_row = sharpened_population[cf_index, :]

        dt_s = time_axis_s[1] - time_axis_s[0]
        total_dur_ms = (time_axis_s[-1] + dt_s) * 1000.0

        if seq_id == silence_seq_id:
            train = np.full(int(round(total_dur_ms)), float(np.mean(sharpened_row)))
        else:
            result, _tone_dur_ms, _isi_ms = chunk_from_id(
                sharpened_row, time_axis_s, seq_id, margin_ms=chunk_margin_ms,
            )
            train = build_prf_boxcar_train(
                [np.mean(chunk) for chunk in result["chunks"]],
                result["onsets_ms"],
                result["offsets_ms"],
                total_dur_ms,
                dt_ms=1.0,
            )
        per_seq[seq_id] = {"train": train}

    return per_seq, cf_hz



def forward_bold_runs(
    per_seq: dict,
    run_designs: list,
    hrf_kernel: np.ndarray,
    cf_hz: float,
    total_run_dur_s: float,
    tr_s: float,
    signal_dt_s: float,
    w: float,
    tau_on_ms: float,
    tau_off_ms: float,
    rectify: bool,
    rho: float,
) -> np.ndarray:
    """Forward-simulate clean BOLD for one parameter combination, all runs.

    The only stage that actually depends on the full
    ``(cf_index via cf_hz/per_seq, alpha via per_seq, tau_on_ms, tau_off_ms,
    w)`` tuple — see module docstring for the caching rationale.

    Parameters
    ----------
    per_seq : dict
        Output of ``build_per_seq_trains`` for this (alpha, cf_index) pair.
    run_designs : list of list of (str, float)
        One stimulus schedule per run (see ``run_assembly.generate_run_design``).
    hrf_kernel : np.ndarray
        Pre-built HRF kernel at ``signal_dt_s`` resolution.
    cf_hz : float
        Characteristic frequency (Hz), for AdapTrans's tau calculation path.
    total_run_dur_s, tr_s, signal_dt_s : float
        Timing parameters, shared across runs.
    w : float
        AdapTrans adaptation weight.
    tau_on_ms, tau_off_ms : float
        AdapTrans ON/OFF filter time constants (ms).
    rectify : bool
        Passed through to ``assemble_run_bold``.
    rho : float
        ON-to-OFF BOLD weighting ratio.

    Returns
    -------
    list of np.ndarray, length ``len(run_designs)``
        Clean (noiseless) BOLD, one array per run — kept unconcatenated
        because ``draw_noisy_measurement`` must add an independent noise
        draw per run before concatenation. Callers wanting a single
        noiseless template vector should ``np.concatenate(...)`` this.
    """
    return [
        assemble_run_bold(
            per_seq=per_seq,
            run_design=run_design,
            total_run_dur_s=total_run_dur_s,
            hrf_kernel=hrf_kernel,
            cf_hz=cf_hz,
            tr_s=tr_s,
            signal_dt_s=signal_dt_s,
            w=w,
            K=None,
            apply_adaptrans_flag=True,
            rectify=rectify,
            rho=rho,
            tau_ms=tau_on_ms,
            tau_ms_off=tau_off_ms,
        )["bold_combined"]
        for run_design in run_designs
    ]


def draw_noisy_measurement(
    bold_clean_per_run: list,
    voxel: str,
    tr_s: float,
    run_seeds: list,
) -> np.ndarray:
    """Add one independent ``PmNoise`` draw per run, then concatenate.

    Parameters
    ----------
    bold_clean_per_run : list of np.ndarray
        Clean BOLD, one array per run (same split used to build
        ``forward_bold_runs``'s concatenated output).
    voxel : str
        ``PmNoise`` preset — ``'low'``, ``'mid'``, or ``'high'``.
    tr_s : float
        Repetition time (s).
    run_seeds : list of int
        One integer seed per run, length ``len(bold_clean_per_run)`` —
        independent across runs and across Monte Carlo repeats (see
        ``spawn_noise_seeds``).

    Returns
    -------
    np.ndarray, shape (n_runs * n_tr_per_run,)
        Noisy BOLD, concatenated across runs in the same order as input.
    """
    noisy_per_run = [
        apply_bold_noise(bold_run, PmNoise(voxel=voxel, seed=seed), tr_s)
        for bold_run, seed in zip(bold_clean_per_run, run_seeds)
    ]
    return np.concatenate(noisy_per_run)


def spawn_noise_seeds(root_seed: int, *entropy: int, n_runs: int) -> list:
    """Derive ``n_runs`` independent, reproducible integer seeds.

    Uses ``np.random.SeedSequence`` keyed on ``(root_seed, *entropy)`` so
    that every Monte Carlo repeat draws genuinely independent noise, while
    the whole experiment is still reproducible from ``root_seed`` alone.
    Mirrors ``tau_recovery_sanity_check.py``'s ``_spawn_run_seeds``.

    Parameters
    ----------
    root_seed : int
        Top-level entropy source.
    *entropy : int
        Additional integer keys identifying this draw (e.g. ground-truth
        combo index, repeat index) — integer-ize any float keys (like ``w``)
        before passing them in.
    n_runs : int
        Number of seeds to return.

    Returns
    -------
    list of int, length ``n_runs``
    """
    root = np.random.SeedSequence([root_seed, *entropy])
    return [int(child.generate_state(1)[0]) for child in root.spawn(n_runs)]
