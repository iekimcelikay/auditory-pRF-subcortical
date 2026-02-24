
import sys
from pathlib import Path
# Ensure project root is on sys.path so tests can import local modules
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np
import soundfile as sf

from save_sound import save_sequence_as_wav


def test_soundsequence_save(out_path):
    """Ensure SoundSequence.save_to_wav writes the correct data."""
    from soundgen import SoundGen


    fs = 100000
    soundgen = SoundGen(fs, 0.005)
    fc = 2500
    num_harmonics=1
    tone_duration=0.267
    harmonic_factor=1
    dbspl=60
    total_duration=5.0
    isi=0.067

    sequence = soundgen.generate_sequence(freq=fc, num_harmonics=num_harmonics, tone_duration=tone_duration,
                        harmonic_factor=harmonic_factor, dbspl=dbspl, total_duration=total_duration, isi=isi)
    num_tones, isi_samples, total_samples = \
            soundgen.calculate_num_tones(tone_duration, isi, total_duration)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    #sequence01_fc440hz_dur200ms_isi100ms_total5sec_numtones1.wav
    out_file = out_path / f"sequence07_fc{fc}hz_dur{tone_duration*1000 :.0f}ms_isi{isi*1000 :.0f}ms_total{total_duration}sec_numtones{num_tones}.wav"

    # Save without normalization so read-back matches closely
    save_sequence_as_wav(sequence, 100000, str(out_file), subtype="FLOAT")

#    data, sr_read = sf.read(str(out_file), dtype="float32")


test_soundsequence_save(str(Path(__file__).parent / "produced"))
