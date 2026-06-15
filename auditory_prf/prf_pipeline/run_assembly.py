"""run_assembly.py
================
Assemble a full-run BOLD timeseries from per-sequence neural responses.

Each unique sequence (WAV file) may appear at multiple onsets within one
fMRI run (repetitions, counterbalanced design).  This module places the
pre-computed pRF-weighted boxcar trains at their respective onsets, then
convolves with the HRF.  An optional AdapTrans step models onset/offset
responses and carry-over adaptation between back-to-back sequences
(controlled by ``apply_adaptrans_flag``).

Works with any pipeline variant (full_pipeline_with_adaptrans,
full_pipeline_shapekern_adaptrans, full_pipeline_impulse_adaptrans, …)
as long as ``per_seq`` contains ``{"train": np.ndarray}`` entries.

Typical usage
-------------
    from auditory_prf.prf_pipeline.full_pipeline_with_adaptrans import run_pipeline`
    from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
    from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold

    seq_results = run_pipeline(exp_name="X")
    hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=1e-3, duration=32.0)

    run_bold = assemble_run_bold(
        per_seq         = seq_results["X"],
        run_design      = [
            ("seq01_fc125hz_dur267ms_isi133ms_...", 10.0),
            ("seq02_fc141hz_dur267ms_isi133ms_...", 20.0),   # back-to-back: no ITI
            (None,                                  30.0),   # null trial
            ("seq01_fc125hz_dur267ms_isi133ms_...", 40.0),   # repetition of seq01
        ],
        total_run_dur_s = 720.0,
        hrf_kernel      = hrf_kernel,
        cf_hz           = 440.0,
    )

Null trials are represented by None or the string "null" in run_design and
contribute zeros to the assembled train.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional, Union

import numpy as np

from auditory_prf.prf_pipeline.adaptrans_onoff_filters import apply_adaptrans
from auditory_prf.prf_pipeline.hrf import convolve_hrf
from auditory_prf.utils.condition_map import make_condition_map, SILENCE_SEQ_ID
from prf_models.pm_noise import PmNoise, apply_bold_noise

logger = logging.getLogger(__name__)


def make_seq_id_fn(
    freq_range: tuple,
    tone_on_ms_options: tuple,
    isi_ms_options: tuple,
    species: str = 'human',
) -> Callable:
    """Build the seq_id mapping function needed by generate_run_design / assemble_run_bold.

    Returns a callable ``(tone_on_ms, isi_ms, freq_hz) -> cond_id str``
    whose output is the stable condition ID used as the universal lookup key
    across WAV files, cochlear NPZ results, and run designs.

    Parameters
    ----------
    freq_range : tuple of (min_hz, max_hz, num_cfs)
        Greenwood CF range — same as used in the experiment and cochlear model.
    tone_on_ms_options : tuple of int
        All tone durations used in the experiment (ms).
    isi_ms_options : tuple of int
        ISI (ms) for each duration — same length as tone_on_ms_options.
    species : str
        'human' or 'cat' (default 'human').
    """
    condition_map = make_condition_map(tone_on_ms_options, isi_ms_options,
                                       freq_range, species=species)

    def _fn(tone_on_ms, isi_ms, freq_hz):
        key = (0, None) if freq_hz is None else (int(tone_on_ms), int(round(freq_hz)))
        cond_id = condition_map.get(key)
        if cond_id is None:
            raise ValueError(
                f"(dur={tone_on_ms} ms, freq={freq_hz:.0f} Hz) not found in condition_map. "
                f"Check tone_on_ms_options and freq_range."
            )
        return cond_id

    return _fn


def assemble_run_bold(
    per_seq: dict,
    run_design: list[tuple[Optional[str], float]],
    total_run_dur_s: float,
    hrf_kernel: np.ndarray,
    cf_hz: float,
    tr_s: float = 1.0,
    signal_dt_s: float = 1e-3,
    w: float = 0.8,
    K: Optional[int] = None,
    apply_adaptrans_flag: bool = True,
    rectify: bool = False,
    rho: float = 1.0,
) -> dict:
    """Assemble a full-run BOLD timeseries from per-stimulus boxcar trains.

    AdapTrans is applied once to the entire assembled neural train so that
    carry-over between back-to-back sequences is modelled correctly.

    Parameters
    ----------
    per_seq : dict
        ``{seq_id: {"train": np.ndarray, ...}}``
        Typically one experiment's entry from ``run_pipeline()`` return value,
        i.e. ``run_pipeline(exp_name)[exp_name]``.
    run_design : list of (seq_id, onset_s)
        Ordered stimulus presentations for one fMRI run.  ``seq_id`` must be
        a key in ``per_seq``; pass ``None`` or ``"null"`` for null trials.
        ``onset_s`` is seconds from run start.
    total_run_dur_s : float
        Full run duration in seconds (e.g. 720.0 for a 12-min run).
    hrf_kernel : np.ndarray, shape (n_kernel,)
        Pre-built HRF kernel at ``signal_dt_s`` resolution.
    cf_hz : float
        Characteristic frequency in Hz used for AdapTrans τ calculation.
        Must match the CF used in ``run_pipeline()``.
    tr_s : float
        TR in seconds.  Output BOLD is downsampled to this rate.
    signal_dt_s : float
        Time resolution of boxcar trains in seconds (default 0.001 = 1 ms).
    w : float
        AdapTrans adaptation weight (default 0.8).
    K : int or None
        AdapTrans kernel length in samples.  Auto-set if None.
    apply_adaptrans_flag : bool
        If False, the assembled boxcar train is used directly as the neural
        signal (no onset/offset decomposition).  Should match the model
        variant used in ``run_pipeline()``.
    rho : float
        ON-to-OFF BOLD weighting ratio.  ``bold_combined = rho * bold_on + bold_off``.
        rho > 1: onset-dominated.  rho = 1: equal weights (default).  rho < 1:
        offset-dominated.  Free parameter during model fitting.

    Returns
    -------
    dict with keys:
        full_train    : np.ndarray, shape (n_1ms,) — assembled pRF-weighted boxcar
        on_response   : np.ndarray, shape (n_1ms,) — AdapTrans ON channel (or full_train)
        off_response  : np.ndarray, shape (n_1ms,) — AdapTrans OFF channel (or zeros)
        bold_on       : np.ndarray, shape (n_TR,)
        bold_off      : np.ndarray, shape (n_TR,)
        bold_combined : np.ndarray, shape (n_TR,)  — rho * bold_on + bold_off
        t_tr          : np.ndarray, shape (n_TR,)  — time axis in seconds
    """
    n_samples = int(round(total_run_dur_s / signal_dt_s))
    full_train = np.zeros(n_samples)

    missing = set()
    for seq_id, onset_s in run_design:
        if seq_id is None or seq_id == "null":
            continue  # legacy fallback; normally null trials use SILENCE_SEQ_ID
        if seq_id not in per_seq:
            missing.add(seq_id)
            continue

        onset_sample = int(round(onset_s / signal_dt_s))
        seq_train = per_seq[seq_id]["train"]

        end   = min(onset_sample + len(seq_train), n_samples)
        n_use = end - onset_sample
        if n_use <= 0:
            logger.warning("seq_id '%s' onset %.1f s is at or past run end (%.1f s). Skipping.",
                           seq_id, onset_s, total_run_dur_s)
            continue
        if n_use < len(seq_train):
            logger.warning("seq_id '%s': stimulus extends %.1f s past run end — truncated.",
                           seq_id, (len(seq_train) - n_use) * signal_dt_s)

        full_train[onset_sample:end] += seq_train[:n_use]

    if missing:
        logger.warning("seq_ids in run_design but not in per_seq: %s", missing)

    # Apply AdapTrans once across the full run
    if apply_adaptrans_flag:
        on_off = apply_adaptrans(
            full_train[np.newaxis, :],
            CFs_Hz=np.array([cf_hz]),
            dt_ms=signal_dt_s * 1000.0,
            w=w,
            K=K,
            pad_value=0.0,
            rectify=rectify,
        )
        on_response  = on_off[0, 0, :]
        off_response = on_off[1, 0, :]
    else:
        on_response  = full_train
        off_response = np.zeros(n_samples)

    bold_on  = convolve_hrf(on_response,  hrf_kernel, signal_dt=signal_dt_s,
                            kernel_dt=signal_dt_s, output_dt=tr_s)
    bold_off = convolve_hrf(off_response, hrf_kernel, signal_dt=signal_dt_s,
                            kernel_dt=signal_dt_s, output_dt=tr_s)

    n_trs = len(bold_on)
    return {
        "full_train":    full_train,
        "on_response":   on_response,
        "off_response":  off_response,
        "bold_on":       bold_on,
        "bold_off":      bold_off,
        "bold_combined": rho * bold_on + bold_off,
        "t_tr":          np.arange(n_trs) * tr_s,
    }


def apply_run_noise(
    bold_combined: np.ndarray,
    noise_model: Optional[PmNoise],
    run_idx: int,
    tr_s: float,
) -> Optional[np.ndarray]:
    """Apply BOLD noise to one run's timeseries, or pass through if no noise model.

    Parameters
    ----------
    bold_combined : np.ndarray, shape (n_tr,)
        Clean BOLD prediction for one run.
    noise_model : PmNoise or None
        Configured noise model. ``None`` means "no noise" — this function
        returns ``None`` in that case so callers can skip saving a noisy
        variant entirely.
    run_idx : int
        0-based run index within the experiment. Used to give each run an
        independent-but-reproducible noise realization (see notes below).
    tr_s : float
        Repetition time in seconds (must match the TR used to produce
        ``bold_combined``).

    Returns
    -------
    np.ndarray, shape (n_tr,), or None
        ``bold_combined`` plus a noise realization, or ``None`` if
        ``noise_model is None``.

    Notes
    -----
    ``PmNoise._make_rng()`` recreates ``np.random.RandomState(seed)`` from
    scratch on every ``compute()`` call when ``seed`` is an integer. Reusing
    one ``PmNoise`` instance across runs with a fixed seed would therefore
    produce an *identical* noise trace for every run. To avoid this, a
    shallow copy of ``noise_model`` is made and its ``seed`` is offset by
    ``run_idx`` before calling ``apply_bold_noise()``. With ``seed='random'``
    or ``seed='none'``, the offset is skipped — each ``compute()`` call
    already draws fresh entropy, or noise is zero.
    """
    if noise_model is None:
        return None
    run_noise = copy.copy(noise_model)
    if isinstance(run_noise.seed, (int, np.integer)):
        run_noise.seed = run_noise.seed + run_idx
    return apply_bold_noise(bold_combined, run_noise, tr_s)


def parse_noise_seed_arg(value: str) -> Union[int, str]:
    """Parse a ``--noise_seed`` CLI string into a ``PmNoise``-compatible seed.

    Parameters
    ----------
    value : str
        Raw CLI argument, e.g. ``"42"``, ``"random"``, or ``"none"``.

    Returns
    -------
    int or str
        ``int(value)`` if ``value`` is an integer string, otherwise ``value``
        unchanged (passed straight through to ``PmNoise(seed=...)``).
    """
    try:
        return int(value)
    except ValueError:
        return value


def generate_run_design(
    base_trials: list,
    seq_id_fn: Callable,
    trial_duration_s: float = 5.0,
    opening_blank_s: float = 10.0,
    iti_range_s: Union[float, tuple] = (1.0, 1.5),
    seed: Optional[int] = None,
) -> list:
    """Generate a run_design without running the PsychoPy experiment.

    Replicates prepare_trials() + _build_timeline() + get_run_design() in
    pure NumPy — no PsychoPy dependency. Use the same seed as the experiment
    for identical trial order.

    Parameters
    ----------
    base_trials : list of (tone_on_ms, isi_ms, freq_hz)
        Master stimulus set (Experiment.base_trials). Null trials have freq_hz=None.
    seq_id_fn : callable
        (tone_on_ms, isi_ms, freq_hz) -> seq_id str | None.
        Build with make_seq_id_fn() from the psychopy module.
    trial_duration_s : float
        Trial duration in seconds (config['trial_duration']).
    opening_blank_s : float
        Opening blank in seconds (config['opening_blank']).
    iti_range_s : tuple of (min_s, max_s)
        Uniform ITI jitter range between trials.
    seed : int or None
        Controls both the shuffle order and ITI jitter for full reproducibility.

    Returns
    -------
    list of (str | None, float)
        run_design ready for assemble_run_bold().
    """
    rng = np.random.default_rng(seed)
    if isinstance(iti_range_s, (int, float)):
        iti_range_s = (iti_range_s, iti_range_s)

    shuffled = list(base_trials)
    rng.shuffle(shuffled)

    # assign onset times with ITI jitter (mirrors _build_timeline)
    run_design = []
    t = opening_blank_s
    for trial in shuffled:
        run_design.append((seq_id_fn(trial[0], trial[1], trial[2]), t))
        t += trial_duration_s + float(rng.uniform(*iti_range_s))

    return run_design


def build_run_design(
    trial_pool: list,
    trial_onsets: list,
    seq_id_fn: Callable,
) -> list:
    """Convert a psychopy trial pool + onset times to run_design for assemble_run_bold.

    Parameters
    ----------
    trial_pool : list of (tone_on_ms, isi_ms, freq_hz, ...)
        Shuffled trial list from ``Experiment.trial_pool``.
    trial_onsets : list of float
        Trial start times in seconds from run start, from ``Experiment._build_timeline()``.
    seq_id_fn : callable
        Maps ``(tone_on_ms, isi_ms, freq_hz)`` → seq_id str, or None for null trials.
        Build with ``make_seq_id_fn()`` in the psychopy module.

    Returns
    -------
    list of (str | None, float)
        Ready to pass as ``run_design`` to ``assemble_run_bold()``.
    """
    return [
        (seq_id_fn(t[0], t[1], t[2]), onset_s)
        for t, onset_s in zip(trial_pool, trial_onsets)
    ]
