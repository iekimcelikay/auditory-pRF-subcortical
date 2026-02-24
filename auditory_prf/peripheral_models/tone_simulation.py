"""
Tone generation simulation using the Zilany2014 cochlea model.

This module handles simulation of tone stimuli generated programmatically.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict


# Project-level imports
from auditory_prf.utils.folder_management import FolderManager
from auditory_prf.stimuli.soundgen import SoundGen
from auditory_prf.utils.stimulus_utils import generate_tone_generator

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.tone_config import ToneConfig
from auditory_prf.peripheral_models.simulation_base import _SimulationBase

logger = logging.getLogger(__name__)


class CochleaSimulation(_SimulationBase):
    """
    Simulation workflow for generated tone stimuli.

    Uses CochleaConfig for model parameters and ToneConfig for stimulus generation.
    """

    def __init__(
            self,
            config: CochleaConfig,
            tone_config: ToneConfig,
            ):

        super().__init__(config)
        self.tone_config = tone_config
        self.name_builder = self._custom_folder_name_builder
        self.sound_gen = SoundGen(self.tone_config.sample_rate, self.tone_config.tau_ramp)
        logger.info("Initialized CochleaSimulation")

    def _custom_folder_name_builder(self, params, timestamp):
        """Build folder name with num_cf, num_ANF, and num_tones."""
        num_cf = params.get('num_cf', 0)
        num_anf = params.get('num_ANF', (0, 0, 0))
        num_tones = params.get('num_tones', 0)

        # Format: results_10cf_32-32-32anf_5tones_20251225_123456
        lsr, msr, hsr = num_anf if isinstance(num_anf, tuple) else (num_anf, num_anf, num_anf)
        return f"results_{num_cf}cf_{lsr}-{msr}-{hsr}anf_{num_tones}tones_{timestamp}"

    def setup_output_folder(self):
        """
        Setup output folder for tone simulation.
        """
        # Use FolderManager to create organized output directory
        folder_manager = FolderManager(
            base_dir = self.config.output_dir,
            name_builder=self.name_builder,
            )

        # Add parameters from both configs for folder name
        folder_params = {
            'num_cf': self.config.num_cf,
            'num_ANF': self.config.num_ANF,
            **self.tone_config.get_folder_params()
        }
        folder_manager.with_params(**folder_params)

        # Create folder (skip stimulus_params.txt - info is in simulation_config)
        self.save_dir = Path(folder_manager.create_folder(save_text=False))
        logger.info(f"Created output directory: {self.save_dir}")

        # Use base class method for logging setup
        self._setup_logging_and_savers()


        # Save configuration metadata (both model and stimulus)
        config_metadata = {
            'processing_type': 'tone_generation',
            # Model parameters
            'peripheral_fs': self.config.peripheral_fs,
            'min_cf': self.config.min_cf,
            'max_cf': self.config.max_cf,
            'num_cf': self.config.num_cf,
            'num_ANF': self.config.num_ANF,
            'anf_types': self.config.anf_types,
            'species': self.config.species,
            'cohc': self.config.cohc,
            'cihc': self.config.cihc,
            'powerlaw': self.config.powerlaw,
            'ffGn': self.config.ffGn,
            'fs_target': self.config.fs_target,
            'seed': self.config.seed,
            # Stimulus parameters
            'sample_rate': self.tone_config.sample_rate,
            'tone_duration': self.tone_config.tone_duration,
            'num_harmonics': self.tone_config.num_harmonics,
            'db_range': self.tone_config.db_range,
            'freq_range': self.tone_config.freq_range,
            }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved cochlea simulation configuration metadata")


    def run(self) -> Dict:
        """
        Run cochlea simulation pipeline.

        Returns:
            Dictionary of lightweight result references (not full data)
        """
        logger.info("=" * 60)
        logger.info("Starting Cochlea Simulation")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup output folder and logging
        if self.save_dir is None:
            self.setup_output_folder()

        # Generate stimuli using tone_generator
        logger.info("Generating stimuli...")
        tone_generator = generate_tone_generator(
            self.sound_gen,
            self.tone_config.db_range,
            self.tone_config.freq_range,
            **self.tone_config.get_stimulus_params()
            )

        # Process through cochlea-zilany2014
        logger.info("Processing through Zilany2014 model...")
        stim_count = 0

        for result in self.processor.process(tone_generator):
            stim_count += 1

            freq = result['freq']
            db = result['db']
            stim_id = f"freq_{freq:.1f}hz_db_{db}"



            stim_data = {
                'freq': freq,
                'db': db,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'time_axis': result['time_axis'] if self.config.save_psth else None,
                'mean_rates': result['mean_rates'] if self.config.save_mean_rates else None,
                'psth': result['psth'] if self.config.save_psth else None,
            }

            # Save immediately (one file per fiber type per stimulus)
            filename = f"{self.config.experiment_name}_{stim_id}"
            self._save_single_result(stim_data, filename)

            # Store lightweight reference only (no large arrays)
            self.results[stim_id] = {
                'freq': freq,
                'db': db,
                'saved_to': filename,
                'cf_list': result['cf_list']
                }

            # CRITICAL: Delete result after all fiber types are saved
            del result, stim_data
            gc.collect()

            logger.info(f"Completed stimulus {stim_count}: {freq:.1f} Hz @ {db} dB (saved & cleared from RAM)")

        # Save runtime info using base class method
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        # Log completion
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results
