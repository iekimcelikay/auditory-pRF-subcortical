"""Render the harmonic FM-sweep stimuli with silences (family P2) to mono 16-bit .wav files.

Design: auditory-prf-docs-all/fm_sweep_silence_kim_design.md. Every stimulus is a harmonic-complex sweep: f0 follows an
ERB-linear triangle over the full 125-800 Hz band (every tone restarts at 125 Hz with a rise), the components are the
harmonics of f0 up to --max-harmonic-hz (SoundGen.harmonic_sweep_maker), and a trial is exactly 20 s of identical
[tone, ISI] blocks that ends with silence. Unit T = 20 s / 120 = 166.67 ms (7350 samples at 44.1 kHz); rates 6, 12, 24 Hz,
so every tone is a whole number of cycles (an even number of passes). Every tone has the same level: its RMS, ramps
included, is the 60 dB SPL RMS of SoundGen.

    harmonic_sweep_armA: tones 1, 2 x T; ISIs 0.5, 1, 2, 4 x T   (3 rates x 2 tones x 4 ISIs = 24 files)
    harmonic_sweep_armB: tones 1, 2, 3, 4 x T; ISIs 1, 2 x T      (3 rates x 4 tones x 2 ISIs = 24 files)

One folder per design, named like the design (stimuli/sweep_design_names.py): <out-root>/<design name>/ holds
    <design name>_<index>_rate<hz>hz_dur<ms>ms_isi<ms>ms.wav     index 01-24: rate first, then tone, then ISI
    silence_20s.wav     sorts last, so it is WAV index 24 of the cochlea SLURM job (slurm_scripts/batch_cochlea_sweep_cfbatch.slurm)
    design_info.json    design name and label, the rendering parameters and the 24 conditions
The names are parsed by examples/cochlea_sweep_wavs.py; the exact tone and ISI lengths (ms in the name are rounded) are
multiples of T and T/2 and are listed in design_info.json.

Usage (from the project root)
-----------------------------
    python stimuli/render_harmonic_sweep_wavs.py                       # both layouts -> stimuli/produced/harmonic_sweep_arm{A,B}/
    python stimuli/render_harmonic_sweep_wavs.py --layout A
    python stimuli/render_harmonic_sweep_wavs.py --out-root /scratch/.../stimuli/produced
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parents[1]))

from stimuli.soundgen import SoundGen
from stimuli.sweep_design_names import HARMONIC_LAYOUT_COUNTS, N_RATES, design_label, harmonic_design_name

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parents[1]))

# ── Design constants (family P2) ──────────────────────────────────────────────
SAMPLE_RATE_HZ = 44100
TRIAL_DURATION_S = 20
TRIAL_SAMPLES = TRIAL_DURATION_S * SAMPLE_RATE_HZ
UNITS_PER_TRIAL = 120
UNIT_SAMPLES = TRIAL_SAMPLES // UNITS_PER_TRIAL             # T = 166.67 ms = 7350 samples
HALF_UNIT_SAMPLES = UNIT_SAMPLES // 2                       # 83.33 ms = 3675 samples
RANGE_LOW_HZ = 125.0
RANGE_HIGH_HZ = 800.0
RATES_HZ = (6.0, 12.0, 24.0)
RAMP_S = 0.010
DBSPL = 60
LAYOUTS = {
    "A": {"tone_units": (1, 2), "isi_half_units": (1, 2, 4, 8)},
    "B": {"tone_units": (1, 2, 3, 4), "isi_half_units": (2, 4)},
}
assert N_RATES == len(RATES_HZ)
for _layout, _spec in LAYOUTS.items():                       # the label counts in sweep_design_names must match
    assert (len(_spec["tone_units"]), len(_spec["isi_half_units"])) == (
        HARMONIC_LAYOUT_COUNTS[_layout]["n_tones"], HARMONIC_LAYOUT_COUNTS[_layout]["n_isis"]), _layout

# Harmonic carrier defaults (listening-chosen, see the design doc)
MAX_HARMONIC_HZ_DEFAULT = 2500.0
TAPER_HZ_DEFAULT = 500.0
SLOPE_DB_PER_OCTAVE_DEFAULT = -6.0
PHASES_DEFAULT = "schroeder"


def hz_to_erb_number(freq_hz):
    """ERB-number (Glasberg & Moore 1990); array in, array out."""
    return 21.4 * np.log10(4.37 * np.asarray(freq_hz, dtype=float) / 1000.0 + 1.0)


def erb_number_to_hz(erb_number):
    """Inverse of hz_to_erb_number."""
    return (10 ** (np.asarray(erb_number, dtype=float) / 21.4) - 1.0) / 4.37 * 1000.0


def triangle_f0_hz(time_in_tone_s: np.ndarray, rate_hz: float) -> np.ndarray:
    """Instantaneous f0 of the symmetric ERB-linear triangle 125 -> 800 -> 125 Hz.

    Parameters
    ----------
    time_in_tone_s : np.ndarray, shape (n_samples,)
        Time since the start of the tone, seconds (the sweep restarts at 125 Hz with a rise at 0).
    rate_hz : float
        Cycles per second (one cycle = one rise + one fall).

    Returns
    -------
    np.ndarray, shape (n_samples,), Hz
    """
    phase = (np.asarray(time_in_tone_s, dtype=float) * rate_hz) % 1.0
    triangle = np.where(phase < 0.5, phase / 0.5, (1.0 - phase) / 0.5)
    low_erb, high_erb = hz_to_erb_number(RANGE_LOW_HZ), hz_to_erb_number(RANGE_HIGH_HZ)
    return erb_number_to_hz(low_erb + (high_erb - low_erb) * triangle)


@functools.lru_cache(maxsize=None)
def reference_rms() -> float:
    """File-unit RMS of a DBSPL tone under SoundGen's calibration (so no calibration constant is duplicated here)."""
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)
    reference = gen.fm_sweep_maker(lambda time_s: np.full_like(time_s, 440.0), 1.0, dbspl=DBSPL)
    return float(np.sqrt(np.mean(reference ** 2)))


def layout_conditions(layout: str, rates_hz: tuple = RATES_HZ) -> list:
    """The 24 conditions of a layout, rate first, then tone, then ISI.

    Returns
    -------
    list of dict, keys: index (1-based), rate_hz, n_tone_samples, n_isi_samples, tone_ms, isi_ms, n_blocks.
    """
    spec = LAYOUTS[layout]
    conditions = []
    for rate_hz in rates_hz:
        for k_tone in spec["tone_units"]:
            for j_isi in spec["isi_half_units"]:
                n_tone, n_isi = k_tone * UNIT_SAMPLES, j_isi * HALF_UNIT_SAMPLES
                n_blocks, remainder = divmod(TRIAL_SAMPLES, n_tone + n_isi)
                assert remainder == 0, f"blocks do not tile the trial: tone {n_tone}, ISI {n_isi}"
                conditions.append({
                    "index": len(conditions) + 1, "rate_hz": rate_hz, "n_tone_samples": n_tone,
                    "n_isi_samples": n_isi, "tone_ms": round(1000 * n_tone / SAMPLE_RATE_HZ),
                    "isi_ms": round(1000 * n_isi / SAMPLE_RATE_HZ), "n_blocks": n_blocks})
    return conditions


def wav_stem(layout: str, condition: dict) -> str:
    """File name without extension, e.g. harmonic_sweep_armB_01_rate6hz_dur167ms_isi167ms."""
    return (f"{harmonic_design_name(layout)}_{condition['index']:02d}_rate{condition['rate_hz']:g}hz"
            f"_dur{condition['tone_ms']}ms_isi{condition['isi_ms']}ms")


def render_tone(
    rate_hz: float,
    n_tone_samples: int,
    max_harmonic_hz: float = MAX_HARMONIC_HZ_DEFAULT,
    taper_hz: float = TAPER_HZ_DEFAULT,
    slope_db_per_octave: float = SLOPE_DB_PER_OCTAVE_DEFAULT,
    phases: str = PHASES_DEFAULT,
) -> np.ndarray:
    """One harmonic-sweep tone with 10 ms sine ramps and RMS (ramps included) equal to the DBSPL RMS.

    Returns
    -------
    np.ndarray, shape (n_tone_samples,), file units (|x| < 1)
    """
    n_passes = 2 * rate_hz * n_tone_samples / SAMPLE_RATE_HZ
    assert abs(n_passes - round(n_passes)) < 1e-9 and round(n_passes) % 2 == 0, \
        f"a tone must be a whole number of cycles, got {n_passes} passes"
    gen = SoundGen(sample_rate=SAMPLE_RATE_HZ, tau=RAMP_S)

    def f0_hz(time_s: np.ndarray) -> np.ndarray:
        return triangle_f0_hz(time_s, rate_hz)

    duration_s = (n_tone_samples + 0.5) / SAMPLE_RATE_HZ     # +0.5 sample: int(fs * duration) cannot floor to n - 1
    raw_tone = gen.harmonic_sweep_maker(f0_hz, duration_s, max_freq_hz=max_harmonic_hz, taper_hz=taper_hz,
                                        slope_db_per_octave=slope_db_per_octave, phases=phases, dbspl=DBSPL)
    tone = gen.sine_ramp(raw_tone)
    assert tone.shape == (n_tone_samples,), (tone.shape, n_tone_samples)
    # the generator calibrates before the ramps; rescale so the tone as played (ramps included) has the target RMS
    return tone * (reference_rms() / np.sqrt(np.mean(tone ** 2)))


def render_trial(condition: dict, **harmonic_kwargs) -> np.ndarray:
    """One 20 s trial: n identical [tone, ISI] blocks, ending with silence.

    Returns
    -------
    np.ndarray, shape (TRIAL_SAMPLES,), file units
    """
    tone = render_tone(condition["rate_hz"], condition["n_tone_samples"], **harmonic_kwargs)
    block = np.concatenate([tone, np.zeros(condition["n_isi_samples"])])
    trial = np.tile(block, condition["n_blocks"])
    assert trial.shape == (TRIAL_SAMPLES,) and trial[-1] == 0.0
    assert np.abs(trial).max() < 1.0, "trial would clip"
    return trial


def write_mono_wav(wav_path: Path, sound: np.ndarray) -> None:
    """Write float file units as mono 16-bit PCM at SAMPLE_RATE_HZ."""
    wavfile.write(wav_path, SAMPLE_RATE_HZ, np.round(sound * 32767).astype(np.int16))


def render_silence_wav(out_dir: Path) -> Path:
    """Write the silent trial (20 s of zeros) as silence_20s.wav (the null-trial response of the run assembly)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"silence_{TRIAL_DURATION_S}s.wav"
    wavfile.write(wav_path, SAMPLE_RATE_HZ, np.zeros(TRIAL_SAMPLES, dtype=np.int16))
    return wav_path


def write_design_info(layout: str, out_dir: Path, rates_hz: tuple, harmonic_params: dict) -> Path:
    """design_info.json: design name and label, rendering parameters and the conditions (exact lengths in samples)."""
    name = harmonic_design_name(layout)
    info = {
        "design": name,
        "label": design_label(name),
        "family": "P2",
        "carrier": "harmonic complex (SoundGen.harmonic_sweep_maker)",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "trial_duration_s": TRIAL_DURATION_S,
        "dbspl": DBSPL,
        "ramp_s": RAMP_S,
        "f0_range_hz": [RANGE_LOW_HZ, RANGE_HIGH_HZ],
        "unit_samples": UNIT_SAMPLES,
        "rates_hz": list(rates_hz),
        **harmonic_params,
        "conditions": [{**condition, "wav": f"{wav_stem(layout, condition)}.wav"}
                       for condition in layout_conditions(layout, rates_hz)],
    }
    info_path = out_dir / "design_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    return info_path


def render_set(
    layout: str,
    out_dir: Path,
    rates_hz: tuple = RATES_HZ,
    write_silence: bool = True,
    max_harmonic_hz: float = MAX_HARMONIC_HZ_DEFAULT,
    taper_hz: float = TAPER_HZ_DEFAULT,
    slope_db_per_octave: float = SLOPE_DB_PER_OCTAVE_DEFAULT,
    phases: str = PHASES_DEFAULT,
) -> list:
    """Render the 24 trials of one layout into out_dir (+ silence_20s.wav and design_info.json).

    Returns
    -------
    list of Path, the written .wav files in sorted (= cochlea array-index) order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    harmonic_params = {"max_harmonic_hz": max_harmonic_hz, "taper_hz": taper_hz,
                       "slope_db_per_octave": slope_db_per_octave, "phases": phases}
    conditions = layout_conditions(layout, rates_hz)
    written = []
    for condition in conditions:
        wav_path = out_dir / f"{wav_stem(layout, condition)}.wav"
        write_mono_wav(wav_path, render_trial(condition, **harmonic_params))
        written.append(wav_path)
        print(f"{condition['index']:2d}/{len(conditions)} {wav_path.name}", flush=True)
    if write_silence:
        written.append(render_silence_wav(out_dir))
    write_design_info(layout, out_dir, rates_hz, harmonic_params)
    return written


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--layout", choices=("A", "B", "both"), default="both")
    parser.add_argument("--out-root", type=str, default=str(PROJECT_ROOT / "stimuli" / "produced"),
                        help="Folder that receives one sub-folder per design, named like the design.")
    parser.add_argument("--max-harmonic-hz", type=float, default=MAX_HARMONIC_HZ_DEFAULT)
    parser.add_argument("--taper-hz", type=float, default=TAPER_HZ_DEFAULT)
    parser.add_argument("--slope-db-per-octave", type=float, default=SLOPE_DB_PER_OCTAVE_DEFAULT)
    parser.add_argument("--phases", choices=("schroeder", "sine"), default=PHASES_DEFAULT,
                        help="Component phases: schroeder (smooth, the chosen default) or all in sine phase.")
    parser.add_argument("--no-silence", action="store_true", help="Do not write silence_20s.wav.")
    args = parser.parse_args(argv)

    for layout in (("A", "B") if args.layout == "both" else (args.layout,)):
        out_dir = Path(args.out_root) / harmonic_design_name(layout)
        print(f"== {harmonic_design_name(layout)}: {design_label(harmonic_design_name(layout))} -> {out_dir}", flush=True)
        written = render_set(layout, out_dir, write_silence=not args.no_silence,
                             max_harmonic_hz=args.max_harmonic_hz, taper_hz=args.taper_hz,
                             slope_db_per_octave=args.slope_db_per_octave, phases=args.phases)
        print(f"{len(written)} wav files in {out_dir}")


if __name__ == "__main__":
    main()
