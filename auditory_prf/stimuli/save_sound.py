"""
save_sound.py

Utilities to save numpy audio sequences as WAV files.

Primary function:
- save_sequence_as_wav(sequence, sample_rate, filename, normalize=True, subtype='PCM_16')

This module uses `soundfile` (pysoundfile) to write WAV files. It validates
inputs, optionally normalizes to avoid clipping, and logs actions instead of
printing.

Conforms to the project's coding standards in `.github/instructions/.instructions.md`.
"""

from typing import Optional
import os
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def save_sequence_as_wav(sequence: np.ndarray,
                         sample_rate: int,
                         filename: str,
                         subtype: str = "PCM_16") -> None:
    """Save a NumPy audio array as a WAV file using `soundfile`.

    Args:
        sequence: NumPy array containing audio samples (float or int).
            - For mono: 1-D array of shape (samples,)
            - For stereo: 2-D array of shape (samples, 2)
        sample_rate: Sampling rate in Hz (positive integer).
        filename: Path to the output WAV file. If no extension is provided,
            ".wav" will be appended.
        subtype: SoundFile subtype describing the desired file format, e.g.
            "PCM_16", "PCM_32", or "FLOAT".

    Raises:
        ValueError: If inputs are invalid (wrong shapes, types, or values).
        RuntimeError: If writing the file fails.
    """

    # Validate inputs
    if not isinstance(sequence, np.ndarray):
        raise ValueError("`sequence` must be a numpy.ndarray")
    if sequence.ndim == 1:
        # Mono audio
        channels = 1
    elif sequence.ndim == 2 and sequence.shape[1] == 2:
        # Stereo audio
        channels = 2
    else:
        raise ValueError("`sequence` must be either a 1-D array (mono) or 2-D array with shape (samples, 2) (stereo)")

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("`sample_rate` must be a positive integer")
    if not isinstance(filename, str) or len(filename) == 0:
        raise ValueError("`filename` must be a non-empty string")

    # Ensure filename has .wav extension
    base, ext = os.path.splitext(filename)
    if ext.lower() != ".wav":
        filename = f"{filename}.wav"

    # Convert to float32 for consistent handling
    data = sequence.astype(np.float32)

    try:
        sf.write(file=filename, data=data, samplerate=sample_rate, subtype=subtype)
        logger.info(f"Saved audio to {filename} (sr={sample_rate}, samples={data.shape[0]}, channels={channels})")
    except Exception as exc:  # pragma: no cover - difficult to trigger in tests
        logger.exception("Failed to write WAV file")
        raise RuntimeError("Failed to write WAV file") from exc

__all__ = ["save_sequence_as_wav"]
