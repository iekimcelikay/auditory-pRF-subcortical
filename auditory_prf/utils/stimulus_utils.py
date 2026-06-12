# stimulus_utils.py
"""
Docstring for utils.stimulus_utils
These calc_cfs functions are COPIED from cochlea toolbox, marek rudnicki.
(Give proper citation).
"""

import numpy as np


def calc_cfs(cf, species):
    if np.isscalar(cf):
        cfs = [float(cf)]

    elif isinstance(cf, tuple) and ('cat' in species):
        # Based on GenerateGreenwood_CFList() from DSAM
        # Liberman (1982)
        aA = 456
        k = 0.8
        a = 2.1

        freq_min, freq_max, freq_num = cf

        xmin = np.log10(freq_min / aA + k) / a
        xmax = np.log10(freq_max / aA + k) / a

        x_map = np.linspace(xmin, xmax, freq_num)
        cfs = aA * (10**(a*x_map) - k)

    elif isinstance(cf, tuple) and ('human' in species):
        # Based on GenerateGreenwood_CFList() from DSAM
        # Liberman (1982)
        aA = 165.4
        k = 0.88
        a = 2.1

        freq_min, freq_max, freq_num = cf

        xmin = np.log10(freq_min / aA + k) / a
        xmax = np.log10(freq_max / aA + k) / a

        x_map = np.linspace(xmin, xmax, freq_num)
        cfs = aA * (10**(a*x_map) - k)

    elif isinstance(cf, list) or isinstance(cf, np.ndarray):
        cfs = cf

    else:
        raise RuntimeError("CF must be a scalar, a tuple or a list.")

    return cfs


def greenwood_human(cf):

    if np.isscalar(cf):
        cfs = [float(cf)]

    elif isinstance(cf, tuple):
        # Based on Greenwood (1990) parameters,
        # function based on 'calc_cfs' from cochlea package.
        aA = 165.4
        k = 0.88
        a = 2.1

        freq_min, freq_max, freq_num = cf

        xmin = np.log10(freq_min / aA + k) / a
        xmax = np.log10(freq_max / aA + k) / a

        x_map = np.linspace(xmin, xmax, freq_num)
        cfs = aA * (10**(a*x_map) - k)

    elif isinstance(cf, list) or isinstance(cf, np.ndarray):
        cfs = cf
    else:
        raise RuntimeError("CF must be a scalar, a tuple or a list.")

    return cfs


def calc_gaussian_centers(
    f_min_hz: float,
    f_max_hz: float,
    n_bands: int,
    scale: str = 'greenwood',
    species: str = 'human',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide [f_min_hz, f_max_hz] into n_bands equal partitions on the chosen
    scale. Return center frequencies and the derived sigma in octaves for each
    band (half-width from center to partition edge).

    Parameters
    ----------
    f_min_hz : float
        Lower edge of the lowest band (Hz).
    f_max_hz : float
        Upper edge of the highest band (Hz).
    n_bands : int
        Number of Gaussian bands.
    scale : {'greenwood', 'erb', 'log'}
        Scale on which bands are equally spaced.
    species : {'human', 'cat'}
        Greenwood species (only used when scale='greenwood').

    Returns
    -------
    centers_hz : np.ndarray, shape (n_bands,)
        Center frequency of each band in Hz.
    sigma_oct : np.ndarray, shape (n_bands,)
        Half-width in octaves from each center to its partition edge.
        Constant across bands for log scale; varies slightly for ERB/Greenwood.
    """
    if scale == 'greenwood':
        if 'human' in species:
            aA, k, a = 165.4, 0.88, 2.1
        elif 'cat' in species:
            aA, k, a = 456.0, 0.80, 2.1
        else:
            raise ValueError(f"Unknown species '{species}' for Greenwood scale.")
        def to_scale(f):
            return np.log10(f / aA + k) / a
        def from_scale(x):
            return aA * (10 ** (a * x) - k)

    elif scale == 'erb':
        # Glasberg & Moore (1990)
        def to_scale(f):
            return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)
        def from_scale(e):
            return (10 ** (e / 21.4) - 1.0) / 4.37 * 1000.0

    elif scale == 'log':
        def to_scale(f):
            return np.log(f)
        def from_scale(x):
            return np.exp(x)

    else:
        raise ValueError(f"Unknown scale '{scale}'. Choose 'greenwood', 'erb', or 'log'.")

    # n_bands+1 equally spaced boundary points on the chosen scale
    boundaries_scale = np.linspace(to_scale(f_min_hz), to_scale(f_max_hz), n_bands + 1)
    boundaries_hz = from_scale(boundaries_scale)

    # Centers are midpoints of adjacent boundaries (in scale space)
    center_scale = (boundaries_scale[:-1] + boundaries_scale[1:]) / 2
    centers_hz = from_scale(center_scale)

    # Sigma = half-width in octaves from center to its upper boundary
    sigma_oct = np.log2(boundaries_hz[1:] / centers_hz)

    return centers_hz, sigma_oct


def generate_stimuli_params(freq_range, db_range):
    """
    Generate stimulus parameters for frequencies and dB levels.

    Args:
        freq_range: Tuple of (min_freq, max_freq, num_freqs) for frequency range
        db_range: Either a list [db1, db2, ...], a tuple (min_db, max_db, step) for range, or a single number

    Returns:
        desired_dbs: Array of dB levels
        desired_freqs: Array of frequencies
    """
    if np.isscalar(db_range):
        desired_dbs = np.array([db_range])
    elif isinstance(db_range, list):
        desired_dbs = np.array(db_range)
    else:
        desired_dbs = np.arange(*db_range)

    desired_freqs = calc_cfs(freq_range, species='human')
    return desired_dbs, desired_freqs


def generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db):
    tone = sound_gen.sound_maker(freq, num_harmonics, duration, harmonic_factor, db)
    ramped_tone = sound_gen.sine_ramp(tone)
    return ramped_tone


def generate_tone_dictionary(
    sound_gen, db_range, freq_range,
    num_tones, num_harmonics, duration, harmonic_factor
):
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range,
                                                         num_tones,
                                                         db_range)
    return {
        (db, freq): generate_ramped_tone(
            sound_gen, freq, num_harmonics, duration, harmonic_factor, db
        )
        for db in desired_dbs for freq in desired_freqs
    }


def generate_tone_generator(
        sound_gen, db_range, freq_range, num_harmonics, duration, harmonic_factor):
    """
    Generate tones on-the-fly using a generator.

    Args:
        sound_gen: SoundGen instance
        db_range: Either a tuple (min_db, max_db, step) or a single dB value
        freq_range: Tuple of (min_freq, max_freq, num_freqs)
        num_harmonics: Number of harmonics in the tone
        duration: Duration of tone in seconds
        harmonic_factor: Harmonic amplitude decay factor

    Yields:
        Tuple of (db, freq, tone)
    """
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range, db_range)
    for db in desired_dbs:
        for freq in desired_freqs:
            # Generate tone only when this iteration happens
            tone = generate_ramped_tone(sound_gen,
                                        freq,
                                        num_harmonics,
                                        duration,
                                        harmonic_factor,
                                        db)
            # Yield tuple of info and tone, so the caller receives it
            yield (db, freq, tone)

def generate_trial_sequences(sound_gen, stimuli, num_harmonics,
                             harmonic_factor, dbspl, total_duration=5.0):

    for tone_on_ms, isi_ms, freq in stimuli:

        if freq is None: # null trial = yield silence, no onsets
            total_samples = int(total_duration * sound_gen.sample_rate)
            sequence = np.zeros((2, total_samples))
            yield (tone_on_ms, isi_ms, freq, sequence, [])
            continue

        sequence, rel_onsets = sound_gen.generate_sequence(
            freq, num_harmonics, tone_on_ms, isi_ms, harmonic_factor,
            dbspl, total_duration)


        yield (tone_on_ms, isi_ms, freq, sequence, rel_onsets)
def ensure_mono(audio, logger):
    """
    Convert stereo audio to mono if needed.

    Parameters
    ----------
    audio : ndarray
        Audio array, either 1D (mono) or 2D (stereo/multichannel)

    Returns
    -------
    ndarray
        1D mono audio array
    """
    if audio.ndim == 2:
        # Average across channels to convert to mono
        audio = audio.mean(axis=1)
        logger.info(f"Converted stereo/multichannel audio to mono by averaging channels")
    return audio

#----------------------------------------