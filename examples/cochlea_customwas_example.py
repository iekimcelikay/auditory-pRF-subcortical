import sys
from pathlib import Path

# Add parent directory to path to import peripheral_models
sys.path.insert(0, str(Path(__file__).parent.parent))

from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulation

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
        num_cf=20,
        num_ANF=(128, 128, 128),
        powerlaw = 'approximate',
        seed = 0,
        fs_target = 1000.0,

        # Output settings
        output_dir = "./models_output/cochlea_wavfiles_120226_05",
        experiment_name="customsequences_test05",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,

        # Logging
        log_console_level = 'INFO',
        log_file_level = 'DEBUG'
        )

    # Get WAV files

    wav_dir = Path("./stimuli/produced")
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    # Run simulation
    simulation = CochleaWavSimulation(config, wav_files, auto_parse=True, parser_func=custom_parser)
    simulation.run()

if __name__ == '__main__':
    main()
