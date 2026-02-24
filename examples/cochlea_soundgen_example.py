from pathlib import Path


from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_simulation import ToneConfig, CochleaSimulation

# ================= Testing ================
# For the soundgen using simulation


if __name__ == '__main__':
    # Model configuration (stimulus-agnostic)
    model_config = CochleaConfig(
        num_cf=3,
        num_ANF=(2,2,2),
        powerlaw= 'approximate',
        output_dir="./models_output/cochlea_test018_12feb_",
        experiment_name="fixedenvironment_test018_meanrate",    # Prefix for all output files

        # Output format control
        save_formats=['npz'],         # Only save NPZ files
        metadata_format='txt',        # Save metadata as text
        save_psth=True,               # Include PSTH arrays
        save_mean_rates=False,         # Include mean rates

        # Logging configuration
        log_level='INFO',             # Overall log level (not used directly)
        log_format='DEFAULT',         # DEFAULT, SIMPLE, or DETAILED
        log_console_level='INFO',     # Console output level
        log_file_level='DEBUG',       # File output level (more detailed)
        log_filename='simulation.log' # Name of log file
        )

    # Tone stimulus configuration
    tone_config = ToneConfig(
        db_range=[60],
        freq_range=(125, 2500, 3),
        tone_duration=0.200,
        num_harmonics=1,
        harmonic_factor=0.5,
        sample_rate=100000,
        tau_ramp=0.005
    )

    # Run simulation with both configs
    simulation = CochleaSimulation(model_config, tone_config)
    results = simulation.run()

    num_tones = tone_config.freq_range[2]  # Number of frequencies from freq_range tuple
    print(f"\nProcessed {len(results)} total results (3 fiber types x {len(tone_config.db_range)} dB levels x {num_tones} tones)")


