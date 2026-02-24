"""
WAV file simulation using analytical mean rate calculation.

This module handles simulation of WAV file stimuli using the rate model
(run_zilany2014_rate) for efficient mean rate computation.
"""

import gc
import logging
from pathlib import Path
import time
from typing import Dict
import soundfile as sf
import thorns.waves as wv


# Project-level imports
from auditory_prf.utils.timestamp_utils import generate_timestamp
from auditory_prf.utils.calculate_population_rate import calculate_population_rate

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.simulation_base import _SimulationBase

logger = logging.getLogger(__name__)


class CochleaWavSimulationMean(_SimulationBase):
    """
    Docstring for CochleaWavSimulation
    """
    @staticmethod
    def parse_wav_filename(filename: str) -> dict:
        """
        Parse filenames like (s3_animal_1_ramp10.wav)
        """

        stem = filename.replace('wav', '')
        parts = stem.split('_')

        if len(parts) >= 4:
            return {
                'sound_number': parts[0],
                'sound_type': parts[1],
                'type_number': parts[2],
                'ramp_id': parts[3]
            }
        # Return empty dict instead of failing
        logger.warning(f"Could not parse metadata from filename: {filename}")
        return {}

    def __init__(
            self,
            config: CochleaConfig,
            wav_files: list,
            metadata_dict: dict = None,
            auto_parse: bool = False,
            parser_func = None
            ):


        super().__init__(config)
        self.wav_files = wav_files

        # Handle metadata
        if metadata_dict is not None:
            # User provided explicit metadata
            self.metadata_dict = metadata_dict
            logger.info(f"Using provided metadata for {len(metadata_dict)} files")

        elif auto_parse:
            # Auto-parse filenames
            parser = parser_func if parser_func is not None else self.parse_wav_filename
            self.metadata_dict = {}

            for wav_file in wav_files:
                identifier = wav_file.stem
                self.metadata_dict[identifier] = parser(wav_file.name)

            logger.info(f"Auto-parsed metadata for {len(self.metadata_dict)} files")

        else:
            # No metadata - just juse empty dicts
            self.metadata_dict = {}
            logger.info("No metadata provided - will save empty metadata dicts")

        logger.info(f"Initialized zilany2014 WavSimulation with {len(wav_files)} files")


    def setup_output_folder(self):
        """Setup output folder - delegates logging to base class."""
        # Create simple folder for WAV files

        timestamp = generate_timestamp()
        lsr, msr, hsr = self.config.num_ANF
        folder_name = f"wav_{self.config.num_cf}cf_analyticalmeanrate_{len(self.wav_files)}files_{timestamp}"

        self.save_dir = Path(self.config.output_dir) / folder_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.save_dir}")

        # Use base class method for logging setup
        self._setup_logging_and_savers()

        # Save configuration metadata (specific to WAV processing)
        config_metadata = {
            'processing_type': 'wav_files',
            'num_files': len(self.wav_files),
            'files': [p.name for p in self.wav_files],
            'peripheral_fs': self.config.peripheral_fs,
            'num_cf': self.config.num_cf,
            'anf_types': self.config.anf_types,
            'species': self.config.species,
            'cohc': self.config.cohc,
            'cihc': self.config.cihc,
        }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved configuration metadata")

    def run(self) -> Dict:
        """Process all WAV files."""
        logger.info("=" * 60)
        logger.info("Starting WAV File Processing")
        logger.info("=" * 60)

        start_time = time.time()

        if self.save_dir is None:
            self.setup_output_folder()

        stim_count = 0

        for wav_path in self.wav_files:
            identifier = wav_path.stem
            logger.info(f"Processing: {wav_path.name}")

            # Load and resample
            audio, fs = sf.read(wav_path)
            if fs != self.config.peripheral_fs:
                logger.info(f"Resampling from {fs} Hz to {self.config.peripheral_fs} Hz")
                audio = wv.resample(audio, fs, self.config.peripheral_fs)

            # Get metadata
            metadata = self.metadata_dict.get(identifier, {})

            # Process using process_wavfile
            result = self.processor.process_wav_meanrate(audio, identifier, metadata)

            # Calculate population rate
            population_rate = calculate_population_rate(result['mean_rates'])

            # Organize results
            result_data = {
                'soundfileid': identifier,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'fiber_types': list(self.config.anf_types),
                'metadata': metadata,
                'population_rate': population_rate,
            }

            if self.config.save_mean_rates:
                result_data['mean_rates'] = result['mean_rates']

            # Note: process_wavfile uses rate model which doesn't produce PSTH
            # time_axis and psth are not available for rate-based processing

            filename = f"{self.config.experiment_name}_{identifier}"
            # Save immediately (use base class method)
            self._save_single_result(result_data, filename)

            # Store lightweight reference only (not full data)
            # This allows iteration but doesn't consume RAM
            self.results[identifier] = {
                'soundfileid': identifier,
                'saved_to': filename,
                'cf_list': result_data['cf_list'],  # Small array, OK to keep
            }

            # CRITICAL: Delete large data immediately after saving
            del result, result_data
            gc.collect()

            stim_count += 1
            logger.info(f"Completed: {identifier} (data saved (including population rates) & cleared from RAM)")

        # Save runtime info (use base class method)
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results
