"""
cochlea_noiseclouds_gaussianprf.py

Runs the Zilany2014 cochlear simulation on one noise-cloud WAV file from the
noiseclouds_gaussianprf stimulus set (produced by
stimuli/save_noise_clouds_gaussian_prf.py).

Designed for SLURM array jobs — pass $SLURM_ARRAY_TASK_ID as --wav-index.

Usage
-----
    # single file (local test)
    python examples/cochlea_noiseclouds_gaussianprf.py --wav-index 0

    # with explicit directory (defaults to latest noiseclouds_gaussianprf_* run)
    python examples/cochlea_noiseclouds_gaussianprf.py \
        --wav-index 5 \
        --wav-dir /scratch/ecelikay/workspace/auditory-pRF-subcortical/stimuli/produced/noiseclouds_gaussianprf_20260602_1915

Filename convention (produced by save_noise_clouds_gaussian_prf.py):
    noisecloud{g:02d}_fc{center}hz_bw{bw}oct_dur{D}ms_isi{I}ms_numtones{N}.wav
    noisecloud00_dur0ms_isi0ms.wav   (silence / null trial)
"""

import argparse
import re
import os
from pathlib import Path

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parent.parent))
STIMULI_PRODUCED = PROJECT_ROOT / "stimuli" / "produced"


# ==============================================================================
# HELPERS
# ==============================================================================

def get_latest_produced_stimuli(prefix: str = "noiseclouds_gaussianprf") -> Path:
    """Return the most recent stimulus directory matching <prefix>_YYYYMMDD_HHMM.

    Directories are sorted lexicographically on the timestamp suffix, which
    works because the format is zero-padded ISO-like (YYYYMMDD_HHMM).

    Parameters
    ----------
    prefix : str
        Directory name prefix to match (default: "noiseclouds_gaussianprf").

    Returns
    -------
    Path
        Path to the latest matching directory.

    Raises
    ------
    FileNotFoundError
        If no matching directories exist under STIMULI_PRODUCED.
    """
    candidates = sorted(STIMULI_PRODUCED.glob(f"{prefix}_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No directories matching '{prefix}_*' found in {STIMULI_PRODUCED}"
        )
    return candidates[-1]


def parse_noisecloud_filename(filename: str) -> dict:
    """Parse a noisecloud filename into metadata fields.

    Handles both stimulus files and the silence null trial.

    Parameters
    ----------
    filename : str
        Basename of the WAV file, e.g.
        ``noisecloud01_fc885hz_bw0.50oct_dur100ms_isi67ms_numtones100.wav``

    Returns
    -------
    dict
        Keys: sequence, center_freq, bandwidth, tone_duration, isi, num_tones.
    """
    stem = filename.replace(".wav", "")
    fc_m    = re.search(r"fc(\d+)hz",        stem)
    bw_m    = re.search(r"bw([\d.]+)oct",    stem)
    dur_m   = re.search(r"dur(\d+)ms",       stem)
    isi_m   = re.search(r"isi(\d+)ms",       stem)
    tones_m = re.search(r"numtones(\d+)",    stem)
    return {
        "sequence":      stem.split("_")[0],
        "center_freq":   f"fc{fc_m.group(1)}hz"       if fc_m    else "fc0hz",
        "bandwidth":     f"bw{bw_m.group(1)}oct"      if bw_m    else "bw0oct",
        "tone_duration": f"dur{dur_m.group(1)}ms"     if dur_m   else "dur0ms",
        "isi":           f"isi{isi_m.group(1)}ms"     if isi_m   else "isi0ms",
        "num_tones":     f"numtones{tones_m.group(1)}" if tones_m else "numtones0",
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cochlear simulation for noiseclouds_gaussianprf stimuli."
    )
    parser.add_argument(
        "--wav-index", type=int, default=0,
        help="Index of WAV file to process. Pass $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--wav-dir", type=str, default=None,
        help="Path to stimulus directory. Defaults to latest noiseclouds_gaussianprf_* run.",
    )
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir) if args.wav_dir else get_latest_produced_stimuli()
    print(f"Stimulus directory : {wav_dir}")

    all_wav_files = sorted(wav_dir.glob("*.wav"))
    if not all_wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    if args.wav_index >= len(all_wav_files):
        print(
            f"wav_index {args.wav_index} out of range — "
            f"only {len(all_wav_files)} files found in {wav_dir}"
        )
        return

    wav_file = [all_wav_files[args.wav_index]]
    print(f"Task {args.wav_index}/{len(all_wav_files) - 1}: {wav_file[0].name}")

    config = CochleaConfig(
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=30,
        num_ANF=(128, 128, 128),
        powerlaw="approximate",
        seed=0,
        fs_target=1000.0,
        output_dir=str(PROJECT_ROOT / "models_output" / wav_dir.name / f"wav{args.wav_index:03d}"),
        experiment_name=f"noiseclouds_gaussianprf_128ANF_wav{args.wav_index:03d}",
        save_formats=["npz"],
        save_mean_rates=False,
        save_psth=True,
        log_console_level="INFO",
        log_file_level="DEBUG",
    )

    simulation = CochleaWavSimulation(
        config, wav_file,
        auto_parse=True,
        parser_func=parse_noisecloud_filename,
    )
    simulation.run()


if __name__ == "__main__":
    main()
