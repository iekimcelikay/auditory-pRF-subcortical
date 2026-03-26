"""
WAV file simulation with full PSTH calculation.

This module handles simulation of WAV file stimuli using the full spike train model
(run_zilany2014) to generate PSTHs along with mean rates.
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
from auditory_prf.utils.stimulus_utils import ensure_mono
from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.simulation_base import _SimulationBase

logger = logging.getLogger(__name__)





class CochleaWavSimulation(_SimulationBase):
    """
    WAV file simulation with full PSTH calculation.

    Uses full spike train model (run_zilany2014) instead of rate model
    to generate PSTHs along with mean rates.
    """

    @staticmethod
    def parse_wav_filename(filename: str) -> dict:
        """
        Parse filenames like (s3_animal_1_ramp10.wav)
        """
        stem = filename.replace('.wav', '')
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
            # No metadata - just use empty dicts
            self.metadata_dict = {}
            logger.info("No metadata provided - will save empty metadata dicts")

        logger.info(f"Initialized zilany2014 WAV PSTH Simulation with {len(wav_files)} files")

    def setup_output_folder(self):
        """Setup output folder for WAV PSTH processing."""
        # Create folder with descriptive name
        timestamp = generate_timestamp()
        lsr, msr, hsr = self.config.num_ANF
        folder_name = f"wav_{self.config.num_cf}cf_{lsr}-{msr}-{hsr}anf_psth_{len(self.wav_files)}files_{timestamp}"

        self.save_dir = Path(self.config.output_dir) / folder_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.save_dir}")

        # Use base class method for logging setup
        self._setup_logging_and_savers()

        # Save configuration metadata
        config_metadata = {
            'processing_type': 'wav_files_psth',
            'num_files': len(self.wav_files),
            'files': [p.name for p in self.wav_files],
            'peripheral_fs': self.config.peripheral_fs,
            'num_cf': self.config.num_cf,
            'num_ANF': self.config.num_ANF,
            'anf_types': self.config.anf_types,
            'species': self.config.species,
            'cohc': self.config.cohc,
            'cihc': self.config.cihc,
            'fs_target': self.config.fs_target,
        }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved configuration metadata")

    def run(self) -> Dict:
        """Process all WAV files with PSTH calculation."""
        logger.info("=" * 60)
        logger.info("Starting WAV File PSTH Processing")
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

            # Ensure mono (convert stereo to mono if needed)
            audio = ensure_mono(audio, logger)

            if fs != self.config.peripheral_fs:
                logger.info(f"Resampling from {fs} Hz to {self.config.peripheral_fs} Hz")
                audio = wv.resample(audio, fs, self.config.peripheral_fs)

            # Get metadata
            metadata = self.metadata_dict.get(identifier, {})

            # Process using process_wav_psth (NEW METHOD)
            result = self.processor.process_wav_psth(audio, identifier, metadata)

            # Calculate population rate from PSTH
            if result['psth'] is not None:
                population_rate_psth = calculate_population_rate(result['psth'])
            else:
                population_rate_psth = None

            # Also calculate from mean rates if available
            if result['mean_rates'] is not None:
                population_rate_mean = calculate_population_rate(result['mean_rates'])
            else:
                population_rate_mean = None

            # Organize results
            result_data = {
                'soundfileid': identifier,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'fiber_types': list(self.config.anf_types),
                'metadata': metadata,
                'time_axis': result['time_axis'] if self.config.save_psth else None,
                'population_rate_psth': population_rate_psth,
                'population_rate_mean': population_rate_mean,
            }

            if self.config.save_mean_rates:
                result_data['mean_rates'] = result['mean_rates']

            if self.config.save_psth:
                result_data['psth'] = result['psth']

            filename = f"{self.config.experiment_name}_{identifier}"
            # Save immediately
            self._save_single_result(result_data, filename)

            # Store lightweight reference only
            self.results[identifier] = {
                'soundfileid': identifier,
                'saved_to': filename,
                'cf_list': result_data['cf_list'],
            }

            # CRITICAL: Delete large data immediately after saving
            del result, result_data
            gc.collect()

            stim_count += 1
            logger.info(f"Completed: {identifier} (PSTH data saved & cleared from RAM)")

        # Save runtime info
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} WAV files with PSTH")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results
