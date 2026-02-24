from pathlib import Path


from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import CochleaWavSimulation


def main():
    """ Process WAV files through cochlea rate model."""

    # Configure the model
    config = CochleaConfig(
        # Model parameters
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=3,
        num_ANF=(2, 2, 2),
        powerlaw = 'approximate',

        # Output settings
        output_dir = "./models_output/cochlea_wavfiles_110226_01",
        experiment_name="jasmin",
        save_formats=['npz'],
        save_mean_rates=True,
        save_psth=False,  # Rate model doesn't produce PSTH

        # Logging
        log_console_level = 'INFO',
        log_file_level = 'DEBUG'
        )

    # Get WAV files

    wav_dir = Path("./stimuli/example_wav")
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    # Run simulation
    simulation = CochleaWavSimulation(config, wav_files, auto_parse=True)
    simulation.run()

if __name__ == '__main__':
    main()
