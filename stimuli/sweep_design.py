"""Design of the FM-sweep (rate-tuning) run: parameters, trial list, rendering.

One run = opening blank + 32 trials x 20 s + closing blank (652.8 s at TR 1.6 s),
matching the tone-cloud design in full_pipeline_toneclouds_adaptrans_claude.py.
Trials are 8 sweep rates x 3 repeats + 8 null trials (NULL_FRACTION = 0.25).
"""

import numpy as np

from auditory_prf.prf_pipeline.run_assembly import generate_run_design
from stimuli.soundgen import SoundGen

# ── Sweep parameters ───────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 44100
RAMP_S = 0.01
TRIAL_DURATION_S = 20.0
SWEEP_DURATION_S = 19.0       # continuous sweep, followed by silence
SILENCE_S = 1.0               # silence at the end of every trial
LOW_FREQ_HZ = 125.0
HIGH_FREQ_HZ = 1600.0
SHAPE = "rounded_triangle"   # 'triangle', 'rounded_triangle' or 'sine'
ROUND_FRACTION = 0.2          # only used by 'rounded_triangle' (tentative)
SPACING = "log"
DBSPL = 60

# Whole cycles per 19 s sweep (0.105 ... 2 Hz); whole counts keep every sweep
# ending exactly at LOW_FREQ_HZ before the 1 s of silence. Roughly geometric,
# 2 cycles (0.105 Hz) to 38 cycles (2.0 Hz, the fastest heard as a clean sweep).
CYCLES_PER_TRIAL = (2, 3, 5, 7, 11, 16, 25, 38)
N_REPEATS = 3
NULL_FRACTION = 0.25

# ── Run timing (same as the tone-cloud design) ─────────────────────────────
TR_S = 1.6
OPENING_BLANK_S = 4 * TR_S
CLOSING_BLANK_S = 4 * TR_S
ITI_RANGE_S = 0


def sweep_rate_hz(n_cycles: int) -> float:
    """Full up-down cycles per second for n_cycles cycles in the 19 s sweep."""
    return n_cycles / SWEEP_DURATION_S


def hz_to_ms(rate_hz: float) -> float:
    """Period in milliseconds of a rate in Hz (e.g. 2 Hz -> 500 ms per cycle)."""
    assert rate_hz > 0, "rate_hz must be positive"
    return 1000.0 / rate_hz


def sweep_seq_id(n_cycles: int, unused_isi_ms: int, sweep_flag) -> str:
    """seq_id for generate_run_design; None marks a null trial.

    Signature mirrors the (tone_on_ms, isi_ms, freq_hz) callable that
    generate_run_design expects: n_cycles takes the tone_on_ms slot and
    sweep_flag (None for null trials) takes the freq_hz slot.
    """
    if sweep_flag is None:
        return None
    return f"fmsweep_{n_cycles:03d}cyc_{sweep_rate_hz(n_cycles):.3f}hz"


def build_base_trials(
    cycles_per_trial: tuple = CYCLES_PER_TRIAL,
    n_repeats: int = N_REPEATS,
    null_fraction: float = NULL_FRACTION,
) -> list:
    """Master trial list: sweep trials plus null trials.

    Returns
    -------
    list of (n_cycles, 0, 1) for sweeps and (0, 0, None) for nulls.
    """
    sweep_trials = [(n_cycles, 0, 1)
                    for n_cycles in cycles_per_trial
                    for _ in range(n_repeats)]
    n_null = int(np.floor(len(sweep_trials) * null_fraction / (1 - null_fraction)))
    return sweep_trials + [(0, 0, None)] * n_null


def build_run_design(seed: int) -> list:
    """Shuffled run design: list of (seq_id | None, onset_s)."""
    run_design = generate_run_design(
        build_base_trials(), sweep_seq_id,
        trial_duration_s=TRIAL_DURATION_S,
        opening_blank_s=OPENING_BLANK_S,
        iti_range_s=ITI_RANGE_S,
        seed=seed,
    )
    return run_design


def total_run_duration_s(run_design: list) -> float:
    """Run length implied by a design: last onset + trial + closing blank."""
    return run_design[-1][1] + TRIAL_DURATION_S + CLOSING_BLANK_S


def render_sweep(n_cycles: int, dbspl: float = DBSPL) -> np.ndarray:
    """Render one 20 s trial: a 19 s continuous sweep, then 1 s of silence.

    Returns
    -------
    np.ndarray, shape (TRIAL_DURATION_S * SAMPLE_RATE_HZ,)
    """
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)
    freq_fn_hz = gen.updown_sweep_fn(LOW_FREQ_HZ, HIGH_FREQ_HZ,
                                     sweep_rate_hz(n_cycles),
                                     SWEEP_DURATION_S, SHAPE, SPACING,
                                     ROUND_FRACTION)
    sound = gen.sine_ramp(gen.fm_sweep_maker(freq_fn_hz, SWEEP_DURATION_S,
                                             dbspl=dbspl))
    trial_sound = np.concatenate([sound, np.zeros(round(SILENCE_S * SAMPLE_RATE_HZ))])
    assert trial_sound.shape == (round(TRIAL_DURATION_S * SAMPLE_RATE_HZ),)
    return trial_sound


# ── Discrete rises and falls (replaces the continuous up-down sweep) ───────
# A trial is repeated blocks of: rise, gap, fall, gap. The frequency range is
# fixed (LOW_FREQ_HZ-HIGH_FREQ_HZ); speed across the 8 conditions is set by
# the sweep duration, i.e. the peak-to-peak repetition rate of the block.
# At SWEEP_S = GAP_S = 0.25 s a block is exactly 1 s (250/250/250/250 ms).
GAP_S = 0.25                  # silence after every rise and every fall
LOUDNESS_COMPENSATION = True  # A-weighting placeholder, not a calibrated curve

# Whole blocks per 20 s trial (roughly geometric, slow -> fast); the sweep
# duration follows from 2 * (sweep_s + GAP_S) = TRIAL_DURATION_S / n_blocks.
BLOCKS_PER_TRIAL = (4, 5, 6, 8, 10, 13, 16, 20)
SWEEP_DURATIONS_S = tuple(TRIAL_DURATION_S / (2 * n_blocks) - GAP_S
                          for n_blocks in BLOCKS_PER_TRIAL)


def a_weighting_db(freq_hz: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting in dB (0 dB at 1 kHz)."""
    freq_sq = np.asarray(freq_hz, dtype=float) ** 2
    linear_gain = (12194.0 ** 2 * freq_sq ** 2) / (
        (freq_sq + 20.6 ** 2)
        * np.sqrt((freq_sq + 107.7 ** 2) * (freq_sq + 737.9 ** 2))
        * (freq_sq + 12194.0 ** 2))
    return 20 * np.log10(linear_gain) + 2.0


def block_duration_s(sweep_s: float, gap_s: float = GAP_S) -> float:
    """Length of one rise, gap, fall, gap block (= peak-to-peak period)."""
    return 2 * (sweep_s + gap_s)


def repetition_rate_hz(sweep_s: float, gap_s: float = GAP_S) -> float:
    """Peak-to-peak repetition rate: blocks per second."""
    return 1.0 / block_duration_s(sweep_s, gap_s)


def sweep_speed_oct_per_s(sweep_s: float) -> float:
    """Speed of one log-spaced rise or fall over the fixed range, in oct/s."""
    return np.log2(HIGH_FREQ_HZ / LOW_FREQ_HZ) / sweep_s


def render_one_sweep(gen: SoundGen, rising: bool, sweep_s: float,
                     dbspl: float = DBSPL,
                     compensate_loudness: bool = LOUDNESS_COMPENSATION
                     ) -> np.ndarray:
    """One ramped log-spaced rise (low->high) or fall (high->low).

    Returns
    -------
    np.ndarray, shape (round(sweep_s * sample_rate),)
    """
    start_hz, end_hz = ((LOW_FREQ_HZ, HIGH_FREQ_HZ) if rising
                        else (HIGH_FREQ_HZ, LOW_FREQ_HZ))

    def freq_fn_hz(time_s):
        return start_hz * (end_hz / start_hz) ** (np.asarray(time_s) / sweep_s)

    plain_sound = gen.fm_sweep_maker(freq_fn_hz, sweep_s, dbspl=dbspl)
    if not compensate_loudness:
        return gen.sine_ramp(plain_sound)
    time_s = np.arange(len(plain_sound)) / gen.sample_rate
    weighted_sound = plain_sound * 10 ** (-a_weighting_db(freq_fn_hz(time_s)) / 20)
    weighted_sound *= np.sqrt(np.mean(plain_sound ** 2)
                              / np.mean(weighted_sound ** 2))   # keep RMS
    return gen.sine_ramp(weighted_sound)


def render_discrete_trial(sweep_s: float, gap_s: float = GAP_S,
                          trial_duration_s: float = TRIAL_DURATION_S
                          ) -> np.ndarray:
    """One 20 s trial of repeated rise, gap, fall, gap blocks.

    Returns
    -------
    np.ndarray, shape (round(trial_duration_s * SAMPLE_RATE_HZ),)
    """
    block_s = block_duration_s(sweep_s, gap_s)
    n_blocks = round(trial_duration_s / block_s)
    assert abs(n_blocks * block_s - trial_duration_s) < 1e-6, (
        f"trial_duration_s={trial_duration_s} is not a whole number of "
        f"{block_s} s blocks")
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)
    rise = render_one_sweep(gen, True, sweep_s)
    fall = render_one_sweep(gen, False, sweep_s)
    trial_sound = np.zeros(round(trial_duration_s * SAMPLE_RATE_HZ))
    for block_idx in range(n_blocks):
        for offset_s, sweep in ((0.0, rise), (sweep_s + gap_s, fall)):
            onset_sample = round((block_idx * block_s + offset_s) * SAMPLE_RATE_HZ)
            trial_sound[onset_sample:onset_sample + len(sweep)] = \
                sweep[:len(trial_sound) - onset_sample]
    assert trial_sound.shape == (round(trial_duration_s * SAMPLE_RATE_HZ),)
    return trial_sound
