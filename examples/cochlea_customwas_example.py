import argparse
from pathlib import Path
import os

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parent.parent))

WAV_DIR_DEFAULT = PROJECT_ROOT / "stimuli" / "produced" / "_20260516_0054"



def custom_parser(filename: str) -> dict:
    """Parse filename: sequence01_fc440hz_dur200ms_isi100ms_total5sec_numtones1.wav"""
    parts = filename.replace('.wav', '').split('_')
    return {
        'sequence': parts[0],
        'center_freq': parts[1],
        'tone_duration': parts[2],
        'isi': parts[3],
        'total_duration': parts[4],
        'num_tones': parts[5],
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
    all_wav_files = sorted(wav_dir.glob("sequence*.wav"))
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
        output_dir=str(PROJECT_ROOT / "models_output" / "dipc_test_20260516_0054"),
        experiment_name=f"dipc_8conditions_isi100ms_128ANF_wav{args.wav_index:03d}",
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
