"""Full spike-train cochlea model (PSTH) for FM-sweep WAVs, one WAV per task.

The WAV folder is stimuli/produced/<design name> (see stimuli/sweep_design_names.py) and the results go to
models_output/<folder name>. Names it understands (parse_sweep_filename):
    harmonic_sweep_armA_01_rate6hz_dur167ms_isi83ms.wav        harmonic sweeps with silences (stimuli/render_harmonic_sweep_wavs.py)
    harmonic_tau_setA_01_f0100hz_dur100ms_isi67ms.wav          static harmonic tones, tau probe (stimuli/render_harmonic_tau_wavs.py)
    fmsweep_fc193.9hz_lo125.0hz_hi276.2hz_010cyc_0.50hz.wav   ERB 3-band design (stimuli/render_sweep_design_erb_wavs.py)
    fmsweep_fc774.6hz_shift0.55oct_017cyc_0.85hz.wav          earlier centered design
    silence_20s.wav                                            the silent trial

Optional CF batching splits the CF axis over several tasks (CFs are independent in
the Zilany model). Each batch writes its own file, suffixed ``_cfb<idx>``; the
batches must be concatenated along the CF axis afterwards.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))      # so that `stimuli` is importable from examples/

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation
from stimuli.sweep_design_names import DEFAULT_DESIGN_NAME, check_design_name, design_label, wav_dir

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parent.parent))
WAV_DIR_DEFAULT = PROJECT_ROOT / wav_dir(DEFAULT_DESIGN_NAME)      # stimuli/produced/sweep_design_erb3band

SWEEP_DURATION_S = 19.9   # sweep part of a 20 s trial (sweep_design_centered.SWEEP_DURATION_S)
SILENCE_S = 0.1           # silence at the end of every trial (sweep_design_centered.SILENCE_S)
TRIAL_DURATION_S = SWEEP_DURATION_S + SILENCE_S
# ERB-band design (stimuli/sweep_design_erb.py): explicit band edges in the name
BAND_NAME_PATTERN = re.compile(
    r"fmsweep_fc(?P<center_hz>\d+(?:\.\d+)?)hz"
    r"_lo(?P<lower_hz>\d+(?:\.\d+)?)hz_hi(?P<upper_hz>\d+(?:\.\d+)?)hz"
    r"_(?P<n_cycles>\d+)cyc"
)
# Old centered design (stimuli/sweep_design_centered.py): geometric center + shift
SWEEP_NAME_PATTERN = re.compile(
    r"fmsweep_fc(?P<center_hz>\d+(?:\.\d+)?)hz"
    r"_shift(?P<shift_oct>\d+(?:\.\d+)?)oct"
    r"_(?P<n_cycles>\d+)cyc"
)


# Silent trial (stimuli/render_sweep_design_erb_wavs.py --silence): silence_20s.wav
SILENCE_NAME_PATTERN = re.compile(r"^silence_(?P<duration_s>\d+)s")

# Harmonic sweeps with silences (stimuli/render_harmonic_sweep_wavs.py, family P2): full-band tones separated by ISIs,
# 20 s trials of identical [tone, ISI] blocks, e.g.
#   harmonic_sweep_armB_01_rate6hz_dur167ms_isi167ms.wav
# dur / isi are the name-rounded milliseconds (tone = 166.67 ms x k, ISI = 83.33 ms x j); the exact values and the
# rendering parameters are in the folder's design_info.json.
TONE_ISI_NAME_PATTERN = re.compile(
    r"^(?P<design>harmonic_sweep_arm[A-Z])_(?P<index>\d{2})"
    r"_rate(?P<rate_hz>\d+(?:\.\d+)?)hz_dur(?P<tone_ms>\d+)ms_isi(?P<isi_ms>\d+)ms"
)
# Static harmonic tones of the tau probe (stimuli/render_harmonic_tau_wavs.py): index 01-24 = f0 (repetition), then Kim
# condition; dur / isi are name-rounded ms, the exact sample counts are in design_info.json, e.g.
#   harmonic_tau_setA_01_f0100hz_dur100ms_isi67ms.wav
TAU_NAME_PATTERN = re.compile(
    r"^(?P<design>harmonic_tau_set[A-Z])_(?P<index>\d{2})"
    r"_f0(?P<f0_hz>\d+(?:\.\d+)?)hz_dur(?P<tone_ms>\d+)ms_isi(?P<isi_ms>\d+)ms"
)
TONE_ISI_TRIAL_DURATION_S = 20.0
TONE_ISI_LOWER_HZ = 125.0     # every tone sweeps the full band (f0 of the harmonic complex)
TONE_ISI_UPPER_HZ = 800.0


def hz_to_erb_number(freq_hz: float) -> float:
    """ERB-number (Glasberg & Moore 1990)."""
    return 21.4 * math.log10(4.37 * freq_hz / 1000.0 + 1.0)


def parse_sweep_filename(filename: str) -> dict:
    """Parse a sweep WAV name into sweep metadata.

    Parameters
    ----------
    filename : str
        ERB-band name ``fmsweep_fc193.9hz_lo125.0hz_hi276.2hz_010cyc_0.50hz.wav``, old
        centered name ``fmsweep_fc774.6hz_shift0.55oct_017cyc_0.85hz.wav``, harmonic-sweep name
        ``harmonic_sweep_armB_01_rate6hz_dur167ms_isi167ms.wav`` (keys: stimulus_type
        "harmonic_sweep_tones", design, design_label, arm, condition_index, carrier, rate_hz,
        tone_ms, isi_ms, lower_hz, upper_hz, width_oct, width_erb, trial_duration_s), or the
        silent trial ``silence_20s.wav``.

    Returns
    -------
    dict
        stimulus_type, center_hz, lower_hz, upper_hz, width_oct, width_erb, n_cycles,
        sweep_duration_s, silence_s, trial_duration_s, rate_hz (plus shift_oct for the
        old naming). rate_hz = n_cycles / sweep_duration_s (exact, not the rounded name
        suffix). ERB names carry lower_hz / upper_hz explicitly; old names give
        center_hz * 2 ** (-/+ shift_oct / 2) (center = geometric center).

    Raises
    ------
    ValueError
        If the name is not a sweep name (no silent placeholder metadata).
    """
    silence_match = SILENCE_NAME_PATTERN.match(filename)
    if silence_match is not None:
        return {"stimulus_type": "silence",
                "trial_duration_s": float(silence_match.group("duration_s")),
                "n_cycles": 0}
    tau_match = TAU_NAME_PATTERN.match(filename)
    if tau_match is not None:
        design = check_design_name(tau_match.group("design"))
        return {
            "stimulus_type": "harmonic_tau_tones",
            "design": design,
            "design_label": design_label(design),
            "set": design[-1],                                      # "A"
            "condition_index": int(tau_match.group("index")),
            "carrier": "harmonic",
            "f0_hz": float(tau_match.group("f0_hz")),
            "tone_ms": float(tau_match.group("tone_ms")),
            "isi_ms": float(tau_match.group("isi_ms")),
            "trial_duration_s": TONE_ISI_TRIAL_DURATION_S,
        }
    tone_isi_match = TONE_ISI_NAME_PATTERN.match(filename)
    if tone_isi_match is not None:
        design = check_design_name(tone_isi_match.group("design"))
        return {
            "stimulus_type": "harmonic_sweep_tones",
            "design": design,
            "design_label": design_label(design),
            "arm": design.rsplit("_", 1)[-1],                      # "armA" / "armB"
            "condition_index": int(tone_isi_match.group("index")),
            "carrier": "harmonic",
            "rate_hz": float(tone_isi_match.group("rate_hz")),
            "tone_ms": float(tone_isi_match.group("tone_ms")),
            "isi_ms": float(tone_isi_match.group("isi_ms")),
            "lower_hz": TONE_ISI_LOWER_HZ,
            "upper_hz": TONE_ISI_UPPER_HZ,
            "width_oct": math.log2(TONE_ISI_UPPER_HZ / TONE_ISI_LOWER_HZ),
            "width_erb": hz_to_erb_number(TONE_ISI_UPPER_HZ) - hz_to_erb_number(TONE_ISI_LOWER_HZ),
            "trial_duration_s": TONE_ISI_TRIAL_DURATION_S,
        }
    band_match = BAND_NAME_PATTERN.search(filename)
    if band_match is not None:
        n_cycles = int(band_match.group("n_cycles"))
        center_hz = float(band_match.group("center_hz"))
        lower_hz = float(band_match.group("lower_hz"))
        upper_hz = float(band_match.group("upper_hz"))
        extra = {}
    else:
        match = SWEEP_NAME_PATTERN.search(filename)
        if match is None:
            raise ValueError(f"not a sweep WAV name: {filename}")
        n_cycles = int(match.group("n_cycles"))
        center_hz = float(match.group("center_hz"))
        shift_oct = float(match.group("shift_oct"))
        lower_hz = center_hz * 2 ** (-0.5 * shift_oct)
        upper_hz = center_hz * 2 ** (0.5 * shift_oct)
        extra = {"shift_oct": shift_oct}
    return {
        "stimulus_type": "fm_sweep",
        "center_hz": center_hz,
        "lower_hz": lower_hz,
        "upper_hz": upper_hz,
        "width_oct": math.log2(upper_hz / lower_hz),
        "width_erb": hz_to_erb_number(upper_hz) - hz_to_erb_number(lower_hz),
        "n_cycles": n_cycles,
        "sweep_duration_s": SWEEP_DURATION_S,
        "silence_s": SILENCE_S,
        "trial_duration_s": TRIAL_DURATION_S,
        "rate_hz": n_cycles / SWEEP_DURATION_S,
        **extra,
    }


def check_trial_length(wav_path: Path, trial_duration_s: float = TRIAL_DURATION_S) -> None:
    """Assert the WAV is the full trial (sweep + trailing silence), not a truncated sweep."""
    info = sf.info(str(wav_path))
    assert abs(info.duration - trial_duration_s) < 1.0 / info.samplerate, (
        f"{wav_path.name}: {info.duration:.4f} s, expected {trial_duration_s} s "
        f"(sweep + {SILENCE_S} s silence)")


def build_config(
    wav_dir: Path,
    wav_index: int,
    cf_batch_size: Optional[int] = None,
    cf_batch_index: int = 0,
    base_seed: int = 0,
    min_cf_hz: float = 125,
    max_cf_hz: float = 2500,
    num_cf: int = 30,
) -> CochleaConfig:
    """CochleaConfig for one WAV, optionally restricted to one CF batch.

    With batching, ``seed = base_seed + cf_batch_index`` so batches do not reuse the
    same random stream. ``num_cf`` stays the full CF count (the batch is a slice of it).
    """
    batch_enabled = cf_batch_size is not None
    batch_suffix = f"_cfb{cf_batch_index:02d}" if batch_enabled else ""
    return CochleaConfig(
        peripheral_fs=100000,
        min_cf=min_cf_hz,
        max_cf=max_cf_hz,
        num_cf=num_cf,
        num_ANF=(128, 128, 128),
        powerlaw='approximate',
        seed=base_seed + (cf_batch_index if batch_enabled else 0),
        fs_target=1000.0,
        cf_batch_enabled=batch_enabled,
        cf_batch_size=cf_batch_size,
        cf_batch_current=cf_batch_index,
        output_dir=str(PROJECT_ROOT / "models_output" / wav_dir.name),
        experiment_name=f"sweep_128ANF_wav{wav_index:03d}{batch_suffix}",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,
        log_console_level='INFO',
        log_file_level='DEBUG',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-index", type=int, default=0,
                        help="Index of WAV file to process. Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--wav-dir", type=str, default=str(WAV_DIR_DEFAULT),
                        help="Path to folder containing WAV files.")
    parser.add_argument("--cf-batch-size", type=int, default=None,
                        help="CFs per batch; omit to process all CFs in one task.")
    parser.add_argument("--cf-batch-index", type=int, default=0,
                        help="Which CF batch (0-based) this task processes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    all_wav_files = sorted(wav_dir.glob("*.wav"))
    if not all_wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    if args.wav_index >= len(all_wav_files):
        print(f"wav_index {args.wav_index} out of range — only {len(all_wav_files)} files found in {wav_dir}")
        return

    wav_file = [all_wav_files[args.wav_index]]
    print(f"Task {args.wav_index}/{len(all_wav_files) - 1}: {wav_file[0].name}")
    check_trial_length(wav_file[0])

    config = build_config(wav_dir, args.wav_index, args.cf_batch_size,
                          args.cf_batch_index, args.seed)
    simulation = CochleaWavSimulation(config, wav_file, auto_parse=True,
                                      parser_func=parse_sweep_filename)
    simulation.run()


if __name__ == '__main__':
    main()
