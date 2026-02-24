from pathlib import Path
import os

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation

PROJECT_ROOT = Path(os.environ.get("AUDITORY_PRF_ROOT", Path(__file__).parent.parent))

def custom_parser(filename:str) -> dict:
    """Parse custon filename format: sequence01_fc440hz_dur200ms_isi100ms_total5sec_numtones1.wav """
    parts = filename.replace('.wav', '').split('_')
    return{
        'sequence': parts[0],
        'center_freq': parts[1],
        'tone_duration': parts[2],
        'isi': parts[3],
        'total_duration': parts[4],
        'num_tones': parts[5],

        }



def main():
    """ Process WAV files through cochlea rate model."""

    # Configure the model
    config = CochleaConfig(
        # Model parameters
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=40,
        num_ANF=(128, 128, 128),
        powerlaw = 'approximate',
        seed = 0,
        fs_target = 1000.0,

        # Output settings
        output_dir = str(PROJECT_ROOT / "models_output" / "dipc_test_240226_02"),
        experiment_name="dipc",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,

        # Logging
        log_console_level = 'INFO',
        log_file_level = 'DEBUG'
        )

    # Get WAV files

    wav_dir = PROJECT_ROOT / "stimuli" / "produced"
    wav_files = sorted(wav_dir.glob("sequence*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    # Run simulation
    simulation = CochleaWavSimulation(config, wav_files, auto_parse=True, parser_func=custom_parser)
    simulation.run()

if __name__ == '__main__':
    main()
