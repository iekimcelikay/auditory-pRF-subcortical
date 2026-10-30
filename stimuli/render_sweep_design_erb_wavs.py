"""Render every stimulus of sweep_design_erb.py to mono 16-bit .wav files.

One folder, len(BANDS_HZ) x len(CYCLES_PER_TRIAL) = 24 files of 20 s at 44.1 kHz
(19.9 s sweep + 100 ms silence). File names are band_seq_id plus the sweep rate, e.g.
fmsweep_fc193.9hz_lo125.0hz_hi276.2hz_010cyc_0.50hz.wav. Sorted order is band first,
then cycle count, which is the array-index order used by the cochlea SLURM job.

Levels follow SoundGen (60 dB SPL calibration, peak ~0.1 of full scale), so
turn the playback volume up.

Usage (from the project root)
-----------------------------
    python stimuli/render_sweep_design_erb_wavs.py --out-dir stimuli/produced/sweep_design_erb_3band
    python stimuli/render_sweep_design_erb_wavs.py --out-dir /path --cycles 199  # 10 Hz only
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parents[1]))

import stimuli.sweep_design_erb as design
from stimuli.sweep_design_centered import SAMPLE_RATE_HZ, TRIAL_DURATION_S


def render_all(out_dir: Path, cycle_counts: tuple = design.CYCLES_PER_TRIAL) -> list:
    """Render all bands x cycle counts into out_dir.

    Returns
    -------
    list of Path, the written .wav files (24 with the default design).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_samples = round(TRIAL_DURATION_S * SAMPLE_RATE_HZ)
    written = []
    for (center_hz, lower_hz, upper_hz), n_cycles in itertools.product(design.BANDS_HZ,
                                                                       cycle_counts):
        trial_sound = design.render_band_sweep(lower_hz, upper_hz, n_cycles)
        assert trial_sound.shape == (expected_samples,)
        assert np.abs(trial_sound).max() < 1.0, "trial would clip"
        wav_path = out_dir / design.wav_name(center_hz, lower_hz, upper_hz, n_cycles)
        wavfile.write(wav_path, SAMPLE_RATE_HZ, (trial_sound * 32767).astype(np.int16))
        written.append(wav_path)
        print(wav_path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--cycles", type=int, nargs="+",
                        default=list(design.CYCLES_PER_TRIAL),
                        help="Cycle counts per 19.9 s sweep (default: the design's 8 counts).")
    args = parser.parse_args()
    written = render_all(Path(args.out_dir), tuple(args.cycles))
    print(f"{len(written)} files in {written[0].parent}")


if __name__ == "__main__":
    main()
