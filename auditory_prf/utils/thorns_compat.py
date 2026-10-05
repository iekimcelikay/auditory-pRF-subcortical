"""Standalone replacements for the `thorns` functions this project uses.

Vendored because `thorns`'s package `__init__.py` unconditionally imports
`thorns.util`, which imports the removed `imp` stdlib module (Python >=3.12)
-- so even `import thorns.waves` fails regardless of which function you need.
These three are copied with no behavior changes from thorns 1.0
(https://github.com/mrkrd/thorns, Marek Rudnicki, GPLv3+).

Original functions: thorns.spikes.trains_to_array, thorns.waves.resample,
thorns.waves.set_dbspl.
"""

import numpy as np
import scipy.signal as dsp


def trains_to_array(spike_trains, fs: float) -> np.ndarray:
    """Convert a spike trains DataFrame to a dense (n_samples, n_fibers) array.

    Parameters
    ----------
    spike_trains : pd.DataFrame
        Must have columns 'spikes' (per-fiber spike times) and 'duration'
        (identical across all rows).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_fibers), spike counts per bin.
    """
    duration = spike_trains['duration'].unique()
    if len(duration) != 1:
        raise ValueError(f"The duration values are not all the same: {duration}")
    duration = duration[0]

    nbins = int(np.ceil(duration * fs))
    tmax = nbins / fs

    signals = []
    for spikes in spike_trains['spikes']:
        signal, _ = np.histogram(spikes, bins=nbins, range=(0, tmax))
        signals.append(signal)

    return np.array(signals).T


def resample(signal: np.ndarray, fs: float, new_fs: float) -> np.ndarray:
    """Resample `signal` from `fs` to `new_fs` Hz."""
    return dsp.resample(signal, int(len(signal) * new_fs / fs))


def set_dbspl(signal: np.ndarray, dbspl: float) -> np.ndarray:
    """Scale `signal` so its RMS corresponds to the given dB SPL."""
    p0 = 20e-6
    rms = np.sqrt(np.sum(signal ** 2) / signal.size)
    return signal * 10 ** (dbspl / 20.0) * p0 / rms
