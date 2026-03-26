"""
Cochlea simulation module - Convenience import module for backward compatibility.

This module re-exports all simulation classes from their respective modules.
The classes have been split into separate files for better code organization:
- simulation_base.py: _SimulationBase (shared logic)
- tone_simulation.py: CochleaSimulation (tone generation)
- wav_simulation_mean.py: CochleaWavSimulationMean (WAV with analytical mean rates)
- wav_simulation_psth.py: CochleaWavSimulation (WAV with full PSTH)

All classes can still be imported from this module for backward compatibility.
"""

# Re-export all classes for backward compatibility
from auditory_prf.peripheral_models.simulation_base import _SimulationBase
from auditory_prf.peripheral_models.tone_simulation import CochleaSimulation
from auditory_prf.peripheral_models.wav_simulation_mean import CochleaWavSimulationMean
from auditory_prf.peripheral_models.wav_simulation_psth import CochleaWavSimulation

# Make classes available when importing *
__all__ = [
    '_SimulationBase',
    'CochleaSimulation',
    'CochleaWavSimulationMean',
    'CochleaWavSimulation',
]

# Import additional dependencies needed for the test block
from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.tone_config import ToneConfig

# ================= Testing ================
# For the soundgen using simulation


if __name__ == '__main__':
    # Model configuration (stimulus-agnostic)
    model_config = CochleaConfig(
        num_cf=10,
        num_ANF=(12,12,12),
        powerlaw= 'approximate',
        output_dir="./models_output/cochlea_test013_approximate",
        experiment_name="test013",    # Prefix for all output files

        # Output format control
        save_formats=['npz'],         # Only save NPZ files
        metadata_format='txt',        # Save metadata as text
        save_psth=True,               # Include PSTH arrays
        save_mean_rates=True,         # Include mean rates

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
        freq_range=(125, 2500, 10),
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


