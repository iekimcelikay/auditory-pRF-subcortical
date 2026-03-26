# ==================== Cochlea Processor ====================

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cochlea.zilany2014.util import calc_cfs
from cochlea.zilany2014 import run_zilany2014
from cochlea.zilany2014 import run_zilany2014_rate


import thorns as th
import thorns.waves as wv

from auditory_prf.peripheral_models.cochlea_config import CochleaConfig
from auditory_prf.utils.calculate_population_rate import calculate_population_rate


logger = logging.getLogger(__name__)

class CochleaProcessor:
    """Process audio stimuli through Cochlea (Zilany 2014) model.

    Generates auditory nerve fiber spike trains using the cochlea package.
    Lazy evaluation - processes one tone at a time.
    """

    def __init__(self, config: CochleaConfig):
        """Initialize processor.

        Args:
            config: CochleaConfig instance with model parameters.
        """
        self.config = config
        self._time_axis = None # Cache time axis

        # Calculate cf_list based on batch mode
        if self.config.cf_batch_enabled and self.config.cf_batch_size is not None:
            # Batch mode: only use CFs for current batch
            self._cf_list = self.config.get_current_batch_cf_array()
            batch_start, batch_end = self.config.get_batch_cf_range(self.config.cf_batch_current)
            logger.info(
                f"CF Batch mode: Processing batch {self.config.cf_batch_current} "
                f"(CF indices {batch_start}-{batch_end}, {len(self._cf_list)} CFs: "
                f"{self._cf_list[0]:.1f} - {self._cf_list[-1]:.1f} Hz)"
                )

        else:
            # Standard mode: Use all CFs
            # Calculate cf_list then to pass to rate function
            self._cf_list = calc_cfs(
                cf=(self.config.min_cf, self.config.max_cf, self.config.num_cf),
                species=self.config.species
                )
            logger.debug(f"CF list: {len(self._cf_list)} CFs from {self._cf_list[0]:.1f} to {self._cf_list[-1]:.1f} Hz")

    # ============ Pipeline Step 1: Run Model ============

    def _run_cochlea_model(self, tone: np.ndarray) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns: ['cf', 'type', 'duration', 'spikes']
        """
        trains = run_zilany2014(
            tone,
            self.config.peripheral_fs,
            **self.config.get_cochlea_kwargs(include_anf_num=True)
            )
        trains.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)
        return trains

    def _run_cochlea_rate_model(self, tone: np.ndarray) -> pd.DataFrame:
        """
        To use the analytical estimation of synaptic rate from synaptic output.

        Returns instantaneous rates at each time sample (peripheral_fs), NOT mean rates
        nor individual ANF responses.

        Args:
            tone: Audio stimulus array

        Returns:
            DataFrame with MultiIndex columns: (anf_type, cf)
            Shape: (time_samples, num_cf * num_anf_types)
            Each column represents one (anf_type, CF) combination

        """
        rates = run_zilany2014_rate(
            tone,
            self.config.peripheral_fs,
            **self.config.get_cochlea_kwargs(include_anf_num=False)
        )

        # rates is a DataFrame with MultiIndex columns: (anf_type, cf)
        # To get mean rates: rates.mean(axis=0) returns Series with MultiIndex

        return rates


    # ============ Pipeline Step 2: Convert Format ==============

    def _convert_to_array(self, trains: pd.DataFrame) -> tuple:
        """Convert spike trains DataFrame to array format."""

        spike_array = th.trains_to_array(trains, self.config.peripheral_fs).T
        cf_list = trains['cf'].unique()
        duration = trains.iloc[0]['duration']
        return spike_array, cf_list, duration

    # ============ Pipeline Step 3: Aggregate by Fiber Type ============

    def _aggregate_by_fiber_type(
            self,
            trains: pd.DataFrame,
            spike_array: np.ndarray,
            cf_list: np.ndarray,
            duration: float
            ) -> tuple:
        """ Aggregate responses by CF and fiber type.

        Returns:
            mean_rates: Dict[fiber_type, array of shape (num_cf,)]
            psth_resampled: Dict[fiber_type, array of shape (num_cf, n_bins)]
            """
        mean_rates = {}
        psth_resampled = {}

        # Calculate dimensions once
        n_cf = len(cf_list)
        bin_samples = int(round(self.config.peripheral_fs / self.config.fs_target))
        n_bins = int(np.floor(spike_array.shape[1] / bin_samples))

        for fiber_type in self.config.anf_types:
            # Filter for this fiber type
            type_mask = trains['type'].values == fiber_type
            type_spike_array = spike_array[type_mask]
            type_cfs = trains.loc[type_mask, 'cf'].values

            # Initialize outputs
            cf_mean_rates = np.zeros(n_cf)
            cf_psth = np.zeros((n_cf, n_bins))

            # Process each CF
            for i_cf, cf in enumerate(cf_list):
                cf_mask = type_cfs == cf
                cf_trains = type_spike_array[cf_mask] # (n_fibers_at_cf, n_samples)

                # Compute mean rate (Hz)
                cf_mean_rates[i_cf] = cf_trains.sum()

                # Average PSTH across fibers, then resample
                avg_psth = cf_trains.mean(axis=0)
                cf_psth[i_cf], self._time_axis = self._resample_psth(
                    avg_psth,
                    self.config.peripheral_fs,
                    self.config.fs_target
                    )

            mean_rates[fiber_type] = cf_mean_rates
            psth_resampled[fiber_type] = cf_psth

        return mean_rates, psth_resampled

    # ============ Pipeline Step 4: Resample PSTH ============

    @staticmethod
    def _resample_psth(spikes: np.ndarray, fs: float, target_fs: float) -> tuple:
        """ Resample PSTH by binning spikes at target sampling rate.

        Args:
            spikes: Spike train array (can be binary or averaged)
            fs: Original sampling freq (Hz)
            target_fs: Target sampling frequency (Hz)

        Returns:
            spike_rates: Array of spike rates at each time bin
            time_axis: Time axis (seconds)
            """
        bin_width_s = 1 / target_fs
        bin_samples = int(round(fs * bin_width_s))
        n_bins = int(np.floor(len(spikes) / bin_samples))

        # Vectorized binning
        spikes_truncated = spikes[:n_bins * bin_samples]
        spike_counts = spikes_truncated.reshape(n_bins, bin_samples).sum(axis=1)
        spike_rates = spike_counts / bin_width_s

        time_axis = np.arange(n_bins) * bin_width_s

        return spike_rates, time_axis

    # ============ Main Processing Loop ============

    def process(self, stimuli):
        """
        Docstring for process

        Args:
            stimuli: Iterator of (db, freq, tone) tuples

        Yields:
            dict: Results for each stimulus containing:
            - db, freq: Stimulus parameters
            - mean_rates: Dict[fiber_type, array(num_cf)]
            - psth: Dict[fiber_type, array(num_cf, n_bins)]
            - cf_list: Array of CFs
            - duration: Stimulus duration
            - time_axis: Time axis for PSTH
        """
        for db, freq, tone in stimuli:
            logger.info(f"Processing {freq:.1f} Hz @ {db} dB")

            try:
                # Pipeline execution
                trains = self._run_cochlea_model(tone)
                spike_array, cf_list, duration = self._convert_to_array(trains)
                mean_rates, psth = self._aggregate_by_fiber_type(
                    trains, spike_array, cf_list, duration
                    )

                yield {
                    'db': db,
                    'freq': freq,
                    'mean_rates': mean_rates,
                    'psth': psth,
                    'cf_list': cf_list,
                    'duration':duration,
                    'time_axis': self._time_axis
                    }

                #Memory cleanup
                del trains, spike_array

            except Exception as e:
                logger.error(f"Failed procesing {freq:.1f}Hz @ {db}dB: {e}")
                raise

    def process_wav_meanrate(self, sound: np.ndarray,
                     identifier: str = "sound",
                        metadata: dict = None):
        """
        Process a wav file through the zilany2014 rate function (analytical estimation)

        Return mean firing rates ( averaged over time ) per CF and fiber type.

        :param self: Description
        :param sound: Description
        :type sound: np.ndarray
        :param identifier: Description
        :type identifier: str
        :param metadata: Description
        :type metadata: dict

        Returns:
            dict: Results containing:
            - mean_rates: Dict[fiber_type, array(num_cf)]
            - cf_list: Array of CFs
            - duration: Sound duration
            - identifier: Sound identifier
            - metadata: Any provided metadata

        """
        logger.info(f"Processing sound: {identifier}")

        try:
            # Get instantaneous rates (returns DataFrame with MultiIndex columns)
            rates_df = self._run_cochlea_rate_model(sound)
            # rates_df columns: MultiIndex[(anf_type, cf), ...]
            # rates_df.shape = (time_samples, num_cf * num_anf_types)

            # Calculate mean rate over time for each channel
            mean_rates_df = rates_df.mean(axis=0)  # Series with MultiIndex

            # Calculate duration from sound array
            duration = len(sound) / self.config.peripheral_fs

            # Organize by fiber type
            mean_rates_dict = {}
            for fiber_type in self.config.anf_types:
                # Extract all CFs for this fiber type
                # MultiIndex selection: (anf_type, cf)
                fiber_rates = mean_rates_df.xs(fiber_type, level='anf_type')
                # Sort by CF to ensure correct order
                fiber_rates = fiber_rates.sort_index()
                mean_rates_dict[fiber_type] = fiber_rates.values

            result = {
                'soundfileid': identifier,
                'mean_rates': mean_rates_dict if self.config.save_mean_rates else None,
                'cf_list': self._cf_list, # Use cached CF list from cochlea
                'duration': duration,
                }

            # Include any provided metadata
            if metadata:
                result['metadata'] = metadata

            return result

        except Exception as e:
            logger.error(f"Failed processing {identifier}: {e}")
            raise

    def process_wav_psth(self, sound: np.ndarray,
                         identifier: str = "sound",
                         metadata: dict = None):
        """
        Process a WAV file through the zilany2014 full spike train model.

        Generates individual spike trains, aggregates them into PSTH, and calculates mean rates.

        Args:
            sound: Audio stimulus array
            identifier: Identifier for this sound
            metadata: Optional metadata dict

        Returns:
            dict: Results containing:
                - mean_rates: Dict[fiber_type, array(num_cf)]
                - psth: Dict[fiber_type, array(num_cf, n_bins)]
                - cf_list: Array of CFs
                - duration: Sound duration
                - time_axis: Time axis for PSTH
                - identifier: Sound identifier
                - metadata: Any provided metadata
        """
        logger.info(f"Processing sound with PSTH: {identifier}")

        try:
            # Run full spike train model (not rate model)
            trains = self._run_cochlea_model(sound)

            # Convert to array format
            spike_array, cf_list, duration = self._convert_to_array(trains)

            # Aggregate by fiber type (gets both mean rates and PSTH)
            mean_rates, psth = self._aggregate_by_fiber_type(
                trains, spike_array, cf_list, duration
            )

            result = {
                'soundfileid': identifier,
                'mean_rates': mean_rates if self.config.save_mean_rates else None,
                'psth': psth if self.config.save_psth else None,
                'cf_list': cf_list,
                'duration': duration,
                'time_axis': self._time_axis
            }

            # Include any provided metadata
            if metadata:
                result['metadata'] = metadata

            return result

        except Exception as e:
            logger.error(f"Failed processing {identifier}: {e}")
            raise

