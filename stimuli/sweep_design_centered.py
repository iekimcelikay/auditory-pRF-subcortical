"""Design of the centered-sweep FM run: 3 centers x 8 cycle counts x 2 shifts.

Each trial is a continuous, log-spaced f0 -> f1 -> f0 asymmetric triangle (rise
takes RISE_FRACTION of a period) repeated for 19.9 s with a whole number of
cycles, then 100 ms of silence. Sweeps are centered (geometrically) on 3 centers
that tile RANGE_LOW_HZ-RANGE_HIGH_HZ when the shift is AUDIBLE_SHIFT_OCT.

One run = opening blank + 32 trials x 20 s + closing blank (652.8 s at TR 1.6 s):
24 sweep trials (3 centers x 8 cycle counts, one each, all at ONE shift) + 8 nulls.
24 runs in total = 4 sessions x 6 runs: 12 runs at the audible shift and 12 at the
inaudible shift, 3 + 3 in every session.
"""

import functools
import itertools
import re

import numpy as np

from auditory_prf.prf_pipeline.run_assembly import generate_run_design
from stimuli.soundgen import SoundGen

# ── Sweep parameters ───────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 44100
RAMP_S = 0.01
DBSPL = 60
TRIAL_DURATION_S = 20.0
SILENCE_S = 0.1                                   # end of every trial
SWEEP_DURATION_S = TRIAL_DURATION_S - SILENCE_S   # 19.9 s
RISE_FRACTION = 0.6                               # share of a period spent rising

RANGE_LOW_HZ = 500.0
RANGE_HIGH_HZ = 1200.0
AUDIBLE_SHIFT_OCT = 0.55       # f1 / f0 = 2 ** shift_oct
INAUDIBLE_SHIFT_OCT = 0.01     # inaudible at all centers and cycle counts (pilot)
N_CENTERS = 3

# Centers are the extreme positions at which a 0.55 oct sweep still stays inside
# the range (605.0 and 991.7 Hz), plus log-spaced centers between them (774.6 Hz).
CENTER_FREQS_HZ = tuple(
    float(center_hz) for center_hz in np.round(np.geomspace(
        RANGE_LOW_HZ * 2 ** (0.5 * AUDIBLE_SHIFT_OCT),
        RANGE_HIGH_HZ / 2 ** (0.5 * AUDIBLE_SHIFT_OCT),
        N_CENTERS), 1))

# Whole cycles per 19.9 s sweep (0.10 ... 2.01 Hz), roughly geometric.
CYCLES_PER_TRIAL = (2, 3, 5, 7, 11, 17, 26, 40)
NULL_FRACTION = 0.25

# ── Run structure ──────────────────────────────────────────────────────────
N_SESSIONS = 4
RUNS_PER_SESSION = 6

# ── Run timing (same as the tone-cloud and rate-tuning designs) ────────────
TR_S = 1.6
OPENING_BLANK_S = 4 * TR_S
CLOSING_BLANK_S = 4 * TR_S
ITI_RANGE_S = 0


def sweep_endpoints_hz(center_hz: float, shift_oct: float) -> tuple:
    """(f0, f1) of the rising half period, symmetric around center_hz in octaves."""
    return center_hz * 2 ** (-0.5 * shift_oct), center_hz * 2 ** (0.5 * shift_oct)


def check_range_coverage(
    center_freqs_hz: tuple = CENTER_FREQS_HZ,
    shift_oct: float = AUDIBLE_SHIFT_OCT,
    range_low_hz: float = RANGE_LOW_HZ,
    range_high_hz: float = RANGE_HIGH_HZ,
    tolerance: float = 1e-3,
) -> None:
    """Assert the sweeps stay inside the range and the bands leave no gap."""
    bands_hz = sorted(sweep_endpoints_hz(center_hz, shift_oct)
                      for center_hz in center_freqs_hz)
    assert bands_hz[0][0] >= range_low_hz * (1 - tolerance), "lowest band below range"
    assert bands_hz[-1][1] <= range_high_hz * (1 + tolerance), "highest band above range"
    assert bands_hz[0][0] <= range_low_hz * (1 + tolerance), "range start not covered"
    assert bands_hz[-1][1] >= range_high_hz * (1 - tolerance), "range end not covered"
    for (_, upper_hz), (lower_hz, _) in zip(bands_hz[:-1], bands_hz[1:]):
        assert lower_hz <= upper_hz, f"gap between {upper_hz:.1f} and {lower_hz:.1f} Hz"


def sweep_speeds_oct_per_s(
    shift_oct: float,
    n_cycles: int,
    rise_fraction: float = RISE_FRACTION,
) -> tuple:
    """(rise, fall) speed in oct/s of a sweep with n_cycles whole cycles in 19.9 s."""
    period_s = SWEEP_DURATION_S / n_cycles
    return (shift_oct / (rise_fraction * period_s),
            shift_oct / ((1.0 - rise_fraction) * period_s))


def centered_seq_id(shift_oct: float, n_cycles: int, unused_isi_ms: int,
                    center_hz) -> str:
    """seq_id for generate_run_design; None marks a null trial.

    Bind shift_oct with functools.partial; the remaining signature mirrors the
    (tone_on_ms, isi_ms, freq_hz) callable that generate_run_design expects:
    n_cycles takes the tone_on_ms slot and center_hz (None for nulls) takes the
    freq_hz slot.
    """
    if center_hz is None:
        return None
    return (f"fmsweep_fc{center_hz:.1f}hz_shift{shift_oct:.2f}oct"
            f"_{n_cycles:03d}cyc")


def parse_seq_id(seq_id: str) -> dict:
    """Inverse of centered_seq_id: center_hz, shift_oct, n_cycles."""
    match = re.fullmatch(r"fmsweep_fc([\d.]+)hz_shift([\d.]+)oct_(\d+)cyc", seq_id)
    assert match is not None, f"not a centered-sweep seq_id: {seq_id}"
    return {"center_hz": float(match.group(1)),
            "shift_oct": float(match.group(2)),
            "n_cycles": int(match.group(3))}


def build_base_trials(
    center_freqs_hz: tuple = CENTER_FREQS_HZ,
    cycles_per_trial: tuple = CYCLES_PER_TRIAL,
    null_fraction: float = NULL_FRACTION,
) -> list:
    """Master trial list for ONE run: every (center, cycle count) once plus nulls.

    Returns
    -------
    list of (n_cycles, 0, center_hz) for sweeps and (0, 0, None) for nulls.
    """
    sweep_trials = [(n_cycles, 0, center_hz)
                    for center_hz, n_cycles in itertools.product(center_freqs_hz,
                                                                 cycles_per_trial)]
    n_null = int(np.floor(len(sweep_trials) * null_fraction / (1 - null_fraction)))
    return sweep_trials + [(0, 0, None)] * n_null


def build_run_design(shift_oct: float, seed: int) -> list:
    """Shuffled run design at one shift: list of (seq_id | None, onset_s)."""
    return generate_run_design(
        build_base_trials(),
        functools.partial(centered_seq_id, shift_oct),
        trial_duration_s=TRIAL_DURATION_S,
        opening_blank_s=OPENING_BLANK_S,
        iti_range_s=ITI_RANGE_S,
        seed=seed,
    )


def total_run_duration_s(run_design: list) -> float:
    """Run length implied by a design: last onset + trial + closing blank."""
    return run_design[-1][1] + TRIAL_DURATION_S + CLOSING_BLANK_S


def shift_for_run(session_idx: int, run_in_session: int) -> float:
    """Shift in octaves (AUDIBLE_SHIFT_OCT or INAUDIBLE_SHIFT_OCT) of one run.

    Parameters
    ----------
    session_idx : int
        0 ... N_SESSIONS - 1.
    run_in_session : int
        0 ... RUNS_PER_SESSION - 1.

    Returns
    -------
    float
        Must give exactly 3 audible and 3 inaudible runs in every session.
    """
    # TODO(human): decide how the two shifts are ordered within a session.
    raise NotImplementedError


def build_all_runs(base_seed: int = 0) -> list:
    """All 24 run designs (4 sessions x 6 runs) with their shift and seed.

    Returns
    -------
    list of dict with keys session_idx, run_in_session, run_idx, shift_oct, seed,
    run_design (list of (seq_id | None, onset_s)).
    """
    runs = []
    for session_idx, run_in_session in itertools.product(range(N_SESSIONS),
                                                         range(RUNS_PER_SESSION)):
        run_idx = session_idx * RUNS_PER_SESSION + run_in_session
        shift_oct = shift_for_run(session_idx, run_in_session)
        seed = base_seed + run_idx
        run_design = build_run_design(shift_oct, seed)
        n_tr = total_run_duration_s(run_design) / TR_S
        assert np.isclose(n_tr, round(n_tr)), f"run {run_idx} is not a whole number of TRs"
        runs.append({"session_idx": session_idx, "run_in_session": run_in_session,
                     "run_idx": run_idx, "shift_oct": shift_oct, "seed": seed,
                     "run_design": run_design})

    for session_idx in range(N_SESSIONS):
        shifts = [run["shift_oct"] for run in runs if run["session_idx"] == session_idx]
        assert shifts.count(AUDIBLE_SHIFT_OCT) == RUNS_PER_SESSION // 2, session_idx
        assert shifts.count(INAUDIBLE_SHIFT_OCT) == RUNS_PER_SESSION // 2, session_idx
    return runs


def triangle_freq_fn_hz(
    center_hz: float,
    period_s: float,
    shift_oct: float,
    rise_fraction: float = RISE_FRACTION,
):
    """Instantaneous frequency of a continuous f0 -> f1 -> f0 log-spaced sweep.

    Returns
    -------
    callable mapping time_s (array) -> freq_hz (array of the same shape)
    """
    assert 0.0 < rise_fraction < 1.0, "rise_fraction must be strictly between 0 and 1"
    start_hz, end_hz = sweep_endpoints_hz(center_hz, shift_oct)

    def freq_fn_hz(time_s):
        phase = (np.asarray(time_s, dtype=float) / period_s) % 1.0
        tri = np.where(phase < rise_fraction,
                       phase / rise_fraction,                    # 0 -> 1 (rise)
                       (1.0 - phase) / (1.0 - rise_fraction))    # 1 -> 0 (fall)
        return start_hz * (end_hz / start_hz) ** tri
    return freq_fn_hz


def render_centered_sweep(center_hz: float, n_cycles: int, shift_oct: float,
                          dbspl: float = DBSPL) -> np.ndarray:
    """One 20 s trial: 19.9 s of repeated rise-fall cycles, then silence.

    Returns
    -------
    np.ndarray, shape (round(TRIAL_DURATION_S * SAMPLE_RATE_HZ),)
    """
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)
    freq_fn_hz = triangle_freq_fn_hz(center_hz, SWEEP_DURATION_S / n_cycles, shift_oct)
    sound = gen.sine_ramp(gen.fm_sweep_maker(freq_fn_hz, SWEEP_DURATION_S, dbspl=dbspl))
    # fm_sweep_maker truncates int(fs * 19.9) = 877589 (float error), so pad the
    # silence to the exact trial length instead of assuming 4410 samples.
    n_trial_samples = round(TRIAL_DURATION_S * SAMPLE_RATE_HZ)
    assert 0 <= n_trial_samples - len(sound) - round(SILENCE_S * SAMPLE_RATE_HZ) <= 1
    trial_sound = np.concatenate([sound, np.zeros(n_trial_samples - len(sound))])
    assert trial_sound.shape == (n_trial_samples,)
    return trial_sound


def render_seq_id(seq_id: str, dbspl: float = DBSPL) -> np.ndarray:
    """Render the trial a seq_id from centered_seq_id names."""
    params = parse_seq_id(seq_id)
    return render_centered_sweep(params["center_hz"], params["n_cycles"],
                                 params["shift_oct"], dbspl)
