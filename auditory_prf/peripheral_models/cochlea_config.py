# CochleaConfig dataclass

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from cochlea.zilany2014.util import calc_cfs


@dataclass
class CochleaConfig:
    """Configuration for Cochlea model parameters - stimulus-agnostic."""

    # ================ Cochlea Model Parameters ================
    peripheral_fs: float = 100000
    min_cf: float = 125
    max_cf: float = 2500
    num_cf: int = 40
    num_ANF: tuple = (128, 128, 128)  # (lsr, msr, hsr)
    anf_types: tuple = ('lsr', 'msr', 'hsr')
    species: str = 'human'
    cohc: float = 1.0
    cihc: float = 1.0
    powerlaw: str = 'approximate'
    ffGn: bool = True
    seed: int = 0  # Random seed for reproducibility
    fs_target: float = 100.0  # Target sampling rate for downsampled PSTH

    # ============ CF Batch Processing ==============
    cf_batch_enabled: bool = False # Enable CF batch processing
    cf_batch_size: Optional[int] = None # Number of CFs per batch (e.g., 10)
    cf_batch_current: int = 0 # Current batch index to process (0-based)

    # ============ Output management ================
    output_dir: str = "./models_output"  # Output under models/output/
    experiment_name: str = "cochlea_experiment"  # Base name for output files
    save_formats: List[str] = field(default_factory=lambda: ['npz', 'pkl', 'mat'])
    metadata_format: str = 'txt'  # 'txt', 'json', or 'yaml'
    save_psth: bool = True
    save_mean_rates: bool = True

    # ============ Logging Configuration ================
    log_level: str = 'INFO'           # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_format: str = 'DEFAULT'       # 'DEFAULT', 'SIMPLE', 'DETAILED'
    log_console_level: str = 'INFO'   # Console output level
    log_file_level: str = 'INFO'      # File output level
    log_filename: str = 'simulation.log'  # Log filename


    def calculate_total_batches(self) -> int:
        """Calculate total number of CF batches.

        Returns:
            Total number of batches needed to cover all CFs
        """
        if not self.cf_batch_enabled or self.cf_batch_size is None:
            return 1
        return int(np.ceil(self.num_cf / self.cf_batch_size))

    def get_batch_cf_range(self, batch_idx: int) -> Tuple[int, int]:
        """Get CF index range for a specific batch.

        Args:
            batch_idx: Batch index (0-based)

        Returns:
            Tuple of (start_idx, end_idx) for CF array slicing
        """

        if not self.cf_batch_enabled or self.cf_batch_size is None:
            return (0, self.num_cf)

        start_idx = batch_idx * self.cf_batch_size
        end_idx = min((batch_idx + 1) * self.cf_batch_size, self.num_cf)
        return (start_idx, end_idx)

    def get_batch_cf_array(self, batch_idx: int) -> np.ndarray:
        """Get CF array for a specific batch.

        Args:
            batch_idx: Batch index (0-based)

        Returns:
            Array of CF values for this batch
        """
        # Calculate full CF list
        cf_list_full = calc_cfs(
            (self.min_cf, self.max_cf, self.num_cf),
            species=self.species
        )

        if not self.cf_batch_enabled or self.cf_batch_size is None:
            return cf_list_full

        # Get batch range
        start_idx, end_idx = self.get_batch_cf_range(batch_idx)
        return cf_list_full[start_idx:end_idx]

    def get_current_batch_cf_array(self) -> np.ndarray:
        """Get CF array for the current batch.

        Returns:
            Array of CF values for current batch (cf_batch_current)
        """
        return self.get_batch_cf_array(self.cf_batch_current)


    def get_cochlea_kwargs(self, include_anf_num: bool = True):
        """Build keyword arguments for zilany2014 functions.

        Args:
            include_anf_num: If True, includes anf_num and seed for run_zilany2014().
                            If False, includes anf_types for run_zilany2014_rate().

        Returns:
            Dictionary with appropriate parameters.
        """

        # Determine CF parameter based on batch mode
        if self.cf_batch_enabled and self.cf_batch_size is not None:
            # Use CF array for current batch (memory efficient!)
            cf_param = self.get_current_batch_cf_array()
        else:
            # Use full range tuple (standard mode)
            cf_param = (self.min_cf, self.max_cf, self.num_cf)


        kwargs = {
            'cf': cf_param,
            'species': self.species,
            'cohc': self.cohc,
            'cihc': self.cihc,
            'powerlaw': self.powerlaw,
            'ffGn': self.ffGn
        }

        if include_anf_num:
            # For run_zilany2014 (spike trains)
            kwargs['anf_num'] = self.num_ANF
            kwargs['seed'] = self.seed
        else:
            # For run_zilany2014_rate (mean rates)
            kwargs['anf_types'] = list(self.anf_types)

        return kwargs