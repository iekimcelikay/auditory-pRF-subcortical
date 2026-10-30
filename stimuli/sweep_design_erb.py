"""ERB-band FM-sweep stimuli: 3 bands tiling 125-800 Hz x 8 cycle counts, no inaudible shift.

The range is split into N_BANDS bands of equal width on the ERB-number scale
(Glasberg & Moore 1990), so every sweep covers the same spectral width
(BAND_WIDTH_ERB) and, at a given rate, the same speed in ERB/s. In octaves the bands
differ (wider at low frequencies). Each trial is a continuous f0 -> f1 -> f0 triangle
that is linear in ERB-number, repeated for 19.9 s with a whole number of cycles, then
100 ms of silence (20 s in total). RISE_FRACTION = 0.5 is the symmetric triangle; the
rise_fraction argument keeps the asymmetric case available.
"""

import re

import numpy as np

from auditory_prf.utils.stimulus_utils import calc_gaussian_centers
from stimuli.soundgen import SoundGen
from stimuli.sweep_design_centered import (
    DBSPL, RAMP_S, SAMPLE_RATE_HZ, SILENCE_S, SWEEP_DURATION_S, TRIAL_DURATION_S)

# ── Sweep parameters ───────────────────────────────────────────────────────
RANGE_LOW_HZ = 125.0
RANGE_HIGH_HZ = 800.0
N_BANDS = 3
RISE_FRACTION = 0.5                     # share of a period spent rising (0.5 = symmetric)

# Whole cycles per 19.9 s sweep (0.50 ... 10.0 Hz), roughly geometric.
CYCLES_PER_TRIAL = (10, 15, 24, 36, 55, 85, 130, 199)


def hz_to_erb_number(freq_hz):
    """ERB-number (Glasberg & Moore 1990) of a frequency in Hz; array in, array out."""
    return 21.4 * np.log10(4.37 * np.asarray(freq_hz, dtype=float) / 1000.0 + 1.0)


def erb_number_to_hz(erb_number):
    """Inverse of hz_to_erb_number."""
    return (10 ** (np.asarray(erb_number, dtype=float) / 21.4) - 1.0) / 4.37 * 1000.0


def calc_bands_hz(
    range_low_hz: float = RANGE_LOW_HZ,
    range_high_hz: float = RANGE_HIGH_HZ,
    n_bands: int = N_BANDS,
) -> tuple:
    """Bands of equal ERB width that tile [range_low_hz, range_high_hz].

    Returns
    -------
    tuple of (center_hz, lower_hz, upper_hz), one per band, lowest first. The center is
    the midpoint in ERB-number (not the geometric center).
    """
    centers_hz, sigma_oct = calc_gaussian_centers(range_low_hz, range_high_hz,
                                                  n_bands, scale='erb')
    upper_hz = centers_hz * 2 ** sigma_oct
    lower_hz = np.concatenate([[range_low_hz], upper_hz[:-1]])
    assert np.isclose(upper_hz[-1], range_high_hz), "bands do not end at range_high_hz"
    return tuple((float(c), float(lo), float(hi))
                 for c, lo, hi in zip(centers_hz, lower_hz, upper_hz))


BANDS_HZ = calc_bands_hz()
BAND_WIDTH_ERB = float((hz_to_erb_number(RANGE_HIGH_HZ) - hz_to_erb_number(RANGE_LOW_HZ))
                       / N_BANDS)


def sweep_speeds_erb_per_s(
    n_cycles: int,
    band_width_erb: float = BAND_WIDTH_ERB,
    rise_fraction: float = RISE_FRACTION,
) -> tuple:
    """(rise, fall) speed in ERB-numbers/s of a sweep with n_cycles whole cycles in 19.9 s."""
    period_s = SWEEP_DURATION_S / n_cycles
    return (band_width_erb / (rise_fraction * period_s),
            band_width_erb / ((1.0 - rise_fraction) * period_s))


def erb_triangle_freq_fn_hz(
    lower_hz: float,
    upper_hz: float,
    period_s: float,
    rise_fraction: float = RISE_FRACTION,
):
    """Instantaneous frequency of a continuous lower -> upper -> lower sweep, linear in ERB.

    Returns
    -------
    callable mapping time_s (array) -> freq_hz (array of the same shape)
    """
    assert 0.0 < rise_fraction < 1.0, "rise_fraction must be strictly between 0 and 1"
    lower_erb = float(hz_to_erb_number(lower_hz))
    upper_erb = float(hz_to_erb_number(upper_hz))

    def freq_fn_hz(time_s):
        phase = (np.asarray(time_s, dtype=float) / period_s) % 1.0
        tri = np.where(phase < rise_fraction,
                       phase / rise_fraction,                    # 0 -> 1 (rise)
                       (1.0 - phase) / (1.0 - rise_fraction))    # 1 -> 0 (fall)
        return erb_number_to_hz(lower_erb + (upper_erb - lower_erb) * tri)
    return freq_fn_hz


def band_seq_id(center_hz: float, lower_hz: float, upper_hz: float, n_cycles: int) -> str:
    """Name of one stimulus, e.g. fmsweep_fc193.9hz_lo125.0hz_hi276.2hz_010cyc."""
    return (f"fmsweep_fc{center_hz:.1f}hz_lo{lower_hz:.1f}hz_hi{upper_hz:.1f}hz"
            f"_{n_cycles:03d}cyc")


def parse_seq_id(seq_id: str) -> dict:
    """Inverse of band_seq_id (the _<rate>hz suffix of a wav name is ignored)."""
    match = re.match(r"fmsweep_fc([\d.]+)hz_lo([\d.]+)hz_hi([\d.]+)hz_(\d+)cyc", seq_id)
    assert match is not None, f"not an ERB-band sweep seq_id: {seq_id}"
    return {"center_hz": float(match.group(1)), "lower_hz": float(match.group(2)),
            "upper_hz": float(match.group(3)), "n_cycles": int(match.group(4))}


def wav_name(center_hz: float, lower_hz: float, upper_hz: float, n_cycles: int) -> str:
    """seq_id plus the sweep rate (cycles per second), e.g. ..._010cyc_0.50hz.wav."""
    seq_id = band_seq_id(center_hz, lower_hz, upper_hz, n_cycles)
    return f"{seq_id}_{n_cycles / SWEEP_DURATION_S:.2f}hz.wav"


def render_band_sweep(lower_hz: float, upper_hz: float, n_cycles: int,
                      rise_fraction: float = RISE_FRACTION,
                      dbspl: float = DBSPL) -> np.ndarray:
    """One 20 s trial: 19.9 s of repeated rise-fall cycles, then 100 ms of silence.

    Returns
    -------
    np.ndarray, shape (round(TRIAL_DURATION_S * SAMPLE_RATE_HZ),)
    """
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)
    freq_fn_hz = erb_triangle_freq_fn_hz(lower_hz, upper_hz, SWEEP_DURATION_S / n_cycles,
                                         rise_fraction)
    sound = gen.sine_ramp(gen.fm_sweep_maker(freq_fn_hz, SWEEP_DURATION_S, dbspl=dbspl))
    # fm_sweep_maker truncates int(fs * 19.9) = 877589 (float error), so pad the
    # silence to the exact trial length instead of assuming 4410 samples.
    n_trial_samples = round(TRIAL_DURATION_S * SAMPLE_RATE_HZ)
    assert 0 <= n_trial_samples - len(sound) - round(SILENCE_S * SAMPLE_RATE_HZ) <= 1
    trial_sound = np.concatenate([sound, np.zeros(n_trial_samples - len(sound))])
    assert trial_sound.shape == (n_trial_samples,)
    return trial_sound
