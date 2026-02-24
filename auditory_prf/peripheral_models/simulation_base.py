"""
Base class for cochlea simulations.

This module contains the shared logic for all simulation types.
"""

import logging
from pathlib import Path
from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.peripheral_models.cochlea_processor import CochleaProcessor
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from auditory_prf.utils.metadata_saver import MetadataSaver
from auditory_prf.utils.result_saver import ResultSaver

logger = logging.getLogger(__name__)


class _SimulationBase:
    """
    Shared logic for simulations using .wav files or generating stimuli.
    Not meant to be used directly.
    """
    def __init__(self, config: CochleaConfig):
        self.config = config
        self.processor = CochleaProcessor(config)
        self.results = {}
        self.save_dir = None
        self.metadata_saver = None
        self.result_saver = None

    def _setup_logging_and_savers(self):
        """ Setup logging and saver instances. """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        format_map = {
            'DEFAULT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'SIMPLE': '%(levelname)s: %(message)s',
            'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            }

        logfile_config = LoggingConfigurator(
            output_dir=self.save_dir,
            log_filename=self.config.log_filename,
            file_level=level_map.get(self.config.log_file_level, logging.INFO),
            console_level=level_map.get(self.config.log_console_level, logging.INFO),
            format_string=format_map.get(self.config.log_format, format_map['DEFAULT'])
        )
        logfile_config.setup()
        logger.info(f"Logging configured: {self.save_dir / self.config.log_filename}")

        self.metadata_saver = MetadataSaver()
        self.result_saver = ResultSaver(self.save_dir)

    def _save_metadata(self, data: dict, base_filename: str):
        """Save metadata. """
        if self.config.metadata_format == 'json':
            self.metadata_saver.save_json(self.save_dir, data, f"{base_filename}.json")
        elif self.config.metadata_format == 'yaml':
            self.metadata_saver.save_yaml(self.save_dir, data, f"{base_filename}.yaml")
        else:
            self.metadata_saver.save_text(self.save_dir, data, f"{base_filename}.txt")

    def _save_single_result(self, data: dict, filename_base: str):
        """Save individual result file (shared code)."""

        if 'npz' in self.config.save_formats:
            self.result_saver.save_npz(data, f"{filename_base}.npz")
        if 'pkl' in self.config.save_formats:
            self.result_saver.save_pickle(data, f"{filename_base}.pkl")
        if 'mat' in self.config.save_formats:
            self.result_saver.save_mat(data, f"{filename_base}.mat")

    def _save_runtime_info(self, elapsed_time: float, stim_count: int):
        """Save runtime info (shared code)."""
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        time_formatted = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s"

        runtime_metadata = {
            'elapsed_time_seconds': round(elapsed_time, 2),
            'elapsed_time_formatted': time_formatted,
            'num_stimuli_processed': stim_count,
            'total_results': len(self.results),
        }
        self._save_metadata(runtime_metadata, 'runtime_info')
