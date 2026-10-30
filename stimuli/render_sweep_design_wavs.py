"""Render every stimulus of sweep_design_centered.py to mono 16-bit .wav files.

One folder per shift (shift_0.55oct/, shift_0.01oct/), each with
len(CENTER_FREQS_HZ) x len(CYCLES_PER_TRIAL) = 24 files of 20 s at 44.1 kHz. The
file names are the seq_ids from centered_seq_id plus the sweep rate in Hz, e.g.
fmsweep_fc774.6hz_shift0.01oct_040cyc_2.01hz.wav. Sorted order is center first, then
cycle count, which is the array-index order used by the cochlea SLURM job.

Levels follow SoundGen (60 dB SPL calibration, peak ~0.1 of full scale), so
turn the playback volume up.

Usage
-----
    python stimuli/render_sweep_design_wavs.py --out-dir /path/to/wavs
    python stimuli/render_sweep_design_wavs.py --out-dir /path/to/wavs --shifts 0.01
    python stimuli/render_sweep_design_wavs.py --out-dir /path/to/wavs --cycles 398  # 20 Hz
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parents[1]))

import stimuli.sweep_design_centered as sweep_design


def wav_name(center_hz: float, shift_oct: float, n_cycles: int) -> str:
    """File name: seq_id plus the sweep rate (cycles per second)."""
    seq_id = sweep_design.centered_seq_id(shift_oct, n_cycles, 0, center_hz)
    return f"{seq_id}_{n_cycles / sweep_design.SWEEP_DURATION_S:.2f}hz.wav"


def render_shift_folder(shift_oct: float, out_dir: Path,
                        cycle_counts: tuple = sweep_design.CYCLES_PER_TRIAL) -> list:
    """Render all centers x cycle counts at one shift into out_dir/shift_<x>oct/.

    Returns
    -------
    list of Path, the written .wav files (24 with the default design).
    """
    shift_dir = out_dir / f"shift_{shift_oct:.2f}oct"
    shift_dir.mkdir(parents=True, exist_ok=True)
    expected_samples = round(sweep_design.TRIAL_DURATION_S * sweep_design.SAMPLE_RATE_HZ)
    written = []
    for center_hz, n_cycles in itertools.product(sweep_design.CENTER_FREQS_HZ,
                                                 cycle_counts):
        trial_sound = sweep_design.render_centered_sweep(center_hz, n_cycles, shift_oct)
        assert trial_sound.shape == (expected_samples,)
        assert np.abs(trial_sound).max() < 1.0, "trial would clip"
        wav_path = shift_dir / wav_name(center_hz, shift_oct, n_cycles)
        wavfile.write(wav_path, sweep_design.SAMPLE_RATE_HZ,
                      (trial_sound * 32767).astype(np.int16))
        written.append(wav_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--shifts", type=float, nargs="+",
                        default=[sweep_design.AUDIBLE_SHIFT_OCT,
                                 sweep_design.INAUDIBLE_SHIFT_OCT],
                        help="Shifts in octaves to render (default: both design shifts).")
    parser.add_argument("--cycles", type=int, nargs="+",
                        default=list(sweep_design.CYCLES_PER_TRIAL),
                        help="Cycle counts per 19.9 s sweep (default: the design's 8 counts).")
    args = parser.parse_args()
    for shift_oct in args.shifts:
        written = render_shift_folder(shift_oct, Path(args.out_dir), tuple(args.cycles))
        print(f"shift {shift_oct:.2f} oct: {len(written)} files in {written[0].parent}")


if __name__ == "__main__":
    main()
