"""condition_map.py
==================
Centralized condition ID convention for the auditory pRF pipeline.

Condition IDs are assigned by sorting all (duration, frequency) pairs:
duration ascending, then frequency ascending within each duration.

The cond_id encodes all timing parameters so it can serve as a universal
lookup key across WAV files, cochlear NPZ results, and run designs, and
so that chunk_from_id can parse dur/isi directly from the key string.

Example with 8 durations × 3 frequencies (ISI=100 ms throughout):
    cond01_fc400hz_dur20ms_isi100ms
    cond02_fc830hz_dur20ms_isi100ms
    cond03_fc1600hz_dur20ms_isi100ms
    cond04_fc400hz_dur30ms_isi100ms
    ...
    cond24_fc1600hz_dur488ms_isi100ms
"""

from auditory_prf.utils.stimulus_utils import calc_cfs
from auditory_prf.utils.timing_utils import fmt_dur_ms, fmt_isi_ms

SILENCE_COND_ID = "cond00"
SILENCE_SEQ_ID  = "cond00_dur0ms_isi0ms"


def make_condition_map(
    tone_on_ms_options: tuple,
    isi_ms_options: tuple,
    freq_range: tuple,
    species: str = 'human',
) -> dict:
    """Return a mapping from (rounded_dur_ms, rounded_freq_hz) to condition ID string.

    Parameters
    ----------
    tone_on_ms_options : tuple of float
        All tone durations used in the experiment (ms).
    isi_ms_options : tuple of float
        ISI (ms) for each duration — same length as tone_on_ms_options.
    freq_range : tuple of (min_hz, max_hz, num_cfs)
        Greenwood CF range — same as used in WAV generation and cochlear model.
    species : str
        'human' or 'cat'.

    Returns
    -------
    dict : {(rounded_dur_ms, rounded_freq_hz): 'cond{N:02d}_fc{F}hz_dur{D}ms_isi{I}ms'}
        Null key (0, None) maps to SILENCE_SEQ_ID.
    """
    desired_freqs = calc_cfs(freq_range, species=species)
    dur_isi_sorted = sorted(zip(tone_on_ms_options, isi_ms_options), key=lambda x: x[0])

    result = {}
    i = 1
    for dur_ms, isi_ms in dur_isi_sorted:
        for freq_hz in desired_freqs:
            key     = (int(round(float(dur_ms))), int(round(freq_hz)))
            cond_id = (f"cond{i:02d}"
                       f"_fc{int(round(freq_hz))}hz"
                       f"_{fmt_dur_ms(dur_ms)}"
                       f"_{fmt_isi_ms(isi_ms)}")
            result[key] = cond_id
            i += 1

    result[(0, None)] = SILENCE_SEQ_ID
    return result