import argparse
from pathlib import Path
import os

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parent.parent))

WAV_DIR_DEFAULT = PROJECT_ROOT / "stimuli" / "produced" / "_20260520_1756"



def custom_parser(filename: str) -> dict:
    """Parse filename: cond01_fc450hz_dur35ms_isi100ms_total20sec_numtones148.wav
    Handles silence file (cond00_dur0ms_isi0ms.wav) which has no fc field.
    """
    import re
    stem = filename.replace('.wav', '')
    fc_m    = re.search(r'fc(\d+)hz',      stem)
    dur_m   = re.search(r'dur(\d+)ms',     stem)
    isi_m   = re.search(r'isi(\d+)ms',     stem)
    total_m = re.search(r'total(\S+?)sec', stem)
    tones_m = re.search(r'numtones(\d+)',  stem)
    return {
        'sequence':       stem.split('_')[0],
        'center_freq':    f"fc{fc_m.group(1)}hz"    if fc_m    else 'fc0hz',
        'tone_duration':  f"dur{dur_m.group(1)}ms"  if dur_m   else 'dur0ms',
        'isi':            f"isi{isi_m.group(1)}ms"  if isi_m   else 'isi0ms',
        'total_duration': f"total{total_m.group(1)}sec" if total_m else 'total0sec',
        'num_tones':      f"numtones{tones_m.group(1)}" if tones_m else 'numtones0',
    }
#correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-index", type=int, default=0,
                        help="Index of WAV file to process. Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--wav-dir", type=str, default=str(WAV_DIR_DEFAULT),
                        help="Path to folder containing WAV files.")
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

    config = CochleaConfig(
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=30,
        num_ANF=(128, 128, 128),
        powerlaw='approximate',
        seed=0,
        fs_target=1000.0,
        output_dir=str(PROJECT_ROOT / "models_output" / wav_dir.name),
        experiment_name=f"dipc_8conditions_isi50ms_128ANF_wav{args.wav_index:03d}",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,
        log_console_level='INFO',
        log_file_level='DEBUG',
    )

    simulation = CochleaWavSimulation(config, wav_file, auto_parse=True, parser_func=custom_parser)
    simulation.run()


if __name__ == '__main__':
    main()
