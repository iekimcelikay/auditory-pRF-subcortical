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