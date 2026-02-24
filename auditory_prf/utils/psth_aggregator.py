"""
PSTH Aggregator - Aggregate results from multiple simulation runs.

Created: 15.02.2026
"""
import os
import numpy as np
from pathlib import Path
import time
from datetime import datetime


class PSTHAggregator:
    """
    Aggregates PSTH results across multiple simulation runs.

    Loads individual run results and computes statistics (mean, std, SEM)
    across the run dimension, providing averaged neural responses.
    """

    def __init__(self, base_output_dir, num_runs):
        """
        Initialize PSTH aggregator.

        Parameters
        ----------
        base_output_dir : str or Path
            Base output directory containing run_XXX subdirectories
        num_runs : int
            Number of runs to aggregate
        """
        self.base_output_dir = Path(base_output_dir)
        self.num_runs = num_runs
        self.aggregated_dir = self.base_output_dir / "aggregated"

        # Create aggregated output directory
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)

        # Track metadata
        self.aggregation_time = None
        self.files_processed = []

    def load_runs_for_file(self, wav_file):
        """
        Load results for a single WAV file across all runs.

        Searches recursively within run directories to handle timestamped subdirectories.

        Parameters
        ----------
        wav_file : Path
            WAV file path

        Returns
        -------
        list of dict
            Loaded data from each run (or None if file missing)
        """
        filename_stem = wav_file.stem
        loaded_data = []

        for run_idx in range(self.num_runs):
            run_dir = self.base_output_dir / f"run_{run_idx:03d}"

            # First try direct path (flat structure)
            npz_file = run_dir / f"{filename_stem}.npz"

            if not npz_file.exists():
                # Search recursively in subdirectories (timestamped structure)
                # Look for files ending with the wav filename stem
                matching_files = list(run_dir.rglob(f"*{filename_stem}.npz"))

                if matching_files:
                    npz_file = matching_files[0]  # Use first match
                else:
                    print(f"Warning: No file matching '{filename_stem}' in {run_dir}")
                    loaded_data.append(None)
                    continue

            data = np.load(npz_file, allow_pickle=True)
            loaded_data.append(dict(data))

        return loaded_data

    def compute_statistics(self, data_list, key):
        """
        Compute statistics across runs for a specific data key.

        Parameters
        ----------
        data_list : list of dict
            Loaded data from each run
        key : str
            Data key to aggregate (e.g., 'population_rate_psth')

        Returns
        -------
        dict
            Statistics: 'mean', 'std', 'sem', 'num_runs'
        """
        # Extract arrays from valid runs
        arrays = []
        for data in data_list:
            if data is not None and key in data:
                arrays.append(data[key])

        if len(arrays) == 0:
            return None

        # Check if all arrays have the same shape
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            print(f"    WARNING: Shape mismatch for '{key}':")
            for i, shape in enumerate(shapes[:5]):  # Show first 5
                print(f"      Run {i}: {shape}")
            if len(shapes) > 5:
                print(f"      ... and {len(shapes) - 5} more runs")

            # Find minimum shape and trim all arrays to match
            min_shape = tuple(min(s) for s in zip(*shapes))
            print(f"    Trimming all arrays to minimum shape: {min_shape}")

            trimmed_arrays = []
            for arr in arrays:
                slices = tuple(slice(0, s) for s in min_shape)
                trimmed_arrays.append(arr[slices])
            arrays = trimmed_arrays

        # Stack along new axis (run dimension)
        stacked = np.stack(arrays, axis=0)  # Shape: (num_runs, ...)

        # Compute statistics
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        sem = std / np.sqrt(len(arrays))

        return {
            'mean': mean,
            'std': std,
            'sem': sem,
            'num_runs': len(arrays)
        }

    def aggregate_file(self, wav_file):
        """
        Aggregate results for a single WAV file across all runs.

        Parameters
        ----------
        wav_file : Path
            WAV file to aggregate

        Returns
        -------
        dict
            Aggregated statistics
        """
        print(f"Aggregating: {wav_file.name}")

        # Load all runs
        data_list = self.load_runs_for_file(wav_file)

        # Determine which keys to aggregate
        valid_data = [d for d in data_list if d is not None]
        if len(valid_data) == 0:
            print(f"  No valid data found for {wav_file.name}")
            return None

        # Keys that represent PSTH data arrays
        psth_keys = ['population_rate_psth', 'psth', 'mean_rates']

        aggregated = {}

        # Aggregate each PSTH key
        for key in psth_keys:
            if key in valid_data[0]:
                print(f"  Aggregating '{key}'...")

                # Handle psth dict (contains hsr, msr, lsr)
                if key == 'psth' and isinstance(valid_data[0][key], (dict, np.ndarray)):
                    # If it's saved as ndarray with item(), extract the dict
                    if isinstance(valid_data[0][key], np.ndarray):
                        psth_data = [d[key].item() if d is not None else None for d in data_list]
                    else:
                        psth_data = [d[key] if d is not None else None for d in data_list]

                    # Aggregate each fiber type
                    aggregated[key] = {}
                    for fiber_type in ['hsr', 'msr', 'lsr']:
                        fiber_arrays = []
                        for psth_dict in psth_data:
                            if psth_dict is not None and fiber_type in psth_dict:
                                fiber_arrays.append(psth_dict[fiber_type])

                        if len(fiber_arrays) > 0:
                            # Check for shape mismatch
                            shapes = [arr.shape for arr in fiber_arrays]
                            if len(set(shapes)) > 1:
                                print(f"    WARNING: Shape mismatch for psth['{fiber_type}']")
                                min_shape = tuple(min(s) for s in zip(*shapes))
                                print(f"    Trimming to: {min_shape}")
                                trimmed = []
                                for arr in fiber_arrays:
                                    slices = tuple(slice(0, s) for s in min_shape)
                                    trimmed.append(arr[slices])
                                fiber_arrays = trimmed

                            stacked = np.stack(fiber_arrays, axis=0)
                            aggregated[key][fiber_type] = {
                                'mean': np.mean(stacked, axis=0),
                                'std': np.std(stacked, axis=0),
                                'sem': np.std(stacked, axis=0) / np.sqrt(len(fiber_arrays))
                            }
                else:
                    # Regular array aggregation
                    stats = self.compute_statistics(data_list, key)
                    if stats is not None:
                        aggregated[key] = stats

        # Copy non-varying metadata from first valid run
        # For time_axis, use the trimmed length if population_rate_psth was trimmed
        metadata_keys = ['cf_list', 'sample_rate', 'metadata']
        for key in metadata_keys:
            if key in valid_data[0]:
                aggregated[key] = valid_data[0][key]

        # Handle time_axis specially - trim to match aggregated data length
        if 'time_axis' in valid_data[0]:
            time_axis = valid_data[0]['time_axis']
            if 'population_rate_psth' in aggregated and isinstance(aggregated['population_rate_psth'], dict):
                # Trimmed - adjust time_axis
                actual_length = aggregated['population_rate_psth']['mean'].shape[-1]
                aggregated['time_axis'] = time_axis[:actual_length]
            else:
                aggregated['time_axis'] = time_axis

        return aggregated

    def save_aggregated(self, aggregated_data, wav_file):
        """
        Save aggregated results to disk in two files:
        1. Main file: Mean values in same format as single runs (compatible)
        2. Stats file: Standard deviation and SEM values

        Parameters
        ----------
        aggregated_data : dict
            Aggregated statistics
        wav_file : Path
            Original WAV file (for naming)
        """
        if aggregated_data is None:
            return

        filename_stem = wav_file.stem

        # File 1: Main aggregated file (mean values, compatible format)
        output_file = self.aggregated_dir / f"{filename_stem}.npz"

        # File 2: Statistics file (std and sem)
        stats_file = self.aggregated_dir / f"{filename_stem}_stats.npz"

        # Prepare main save dict (compatible format with single runs)
        save_dict = {}
        stats_dict = {}

        # Add aggregated data
        for key, value in aggregated_data.items():
            if key == 'psth' and isinstance(value, dict):
                # Handle psth dict with fiber types
                # Main file: mean values in same nested structure
                psth_mean = {}
                psth_std = {}
                psth_sem = {}

                for fiber_type, stats in value.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        psth_mean[fiber_type] = stats['mean']
                        psth_std[fiber_type] = stats['std']
                        psth_sem[fiber_type] = stats['sem']

                # Save as object arrays to preserve dict structure
                save_dict[key] = np.array(psth_mean, dtype=object)
                stats_dict[f"{key}_std"] = np.array(psth_std, dtype=object)
                stats_dict[f"{key}_sem"] = np.array(psth_sem, dtype=object)

            elif isinstance(value, dict) and 'mean' in value:
                # Regular arrays: save mean in main, std/sem in stats
                save_dict[key] = value['mean']
                stats_dict[f"{key}_std"] = value['std']
                stats_dict[f"{key}_sem"] = value['sem']
                if 'num_runs' in value:
                    save_dict['num_runs'] = value['num_runs']
            else:
                # Copy metadata as-is to main file
                save_dict[key] = value

        # Add aggregation metadata to main file
        save_dict['aggregation_metadata'] = {
            'num_runs': self.num_runs,
            'aggregation_time': datetime.now().isoformat(),
            'base_output_dir': str(self.base_output_dir),
            'is_aggregated': True,
            'stats_file': f"{filename_stem}_stats.npz"
        }

        # Save both files
        np.savez_compressed(output_file, **save_dict)
        print(f"  Saved (mean): {output_file}")

        if stats_dict:
            stats_dict['aggregation_metadata'] = save_dict['aggregation_metadata']
            np.savez_compressed(stats_file, **stats_dict)
            print(f"  Saved (std/sem): {stats_file}")

    def aggregate_all_files(self, wav_files):
        """
        Aggregate results for all WAV files.

        Parameters
        ----------
        wav_files : list of Path
            All WAV files to aggregate

        Returns
        -------
        list of Path
            Paths to saved aggregated files
        """
        print(f"\n{'='*60}")
        print(f"Aggregating Results")
        print(f"{'='*60}")
        print(f"Number of files: {len(wav_files)}")
        print(f"Number of runs per file: {self.num_runs}")
        print(f"Output directory: {self.aggregated_dir}")
        print(f"{'='*60}\n")

        self.aggregation_time = time.time()
        aggregated_files = []

        for wav_file in wav_files:
            aggregated_data = self.aggregate_file(wav_file)
            if aggregated_data is not None:
                self.save_aggregated(aggregated_data, wav_file)
                self.files_processed.append(wav_file.name)
                aggregated_files.append(
                    self.aggregated_dir / f"{wav_file.stem}.npz"
                )

        elapsed = time.time() - self.aggregation_time

        print(f"\n{'='*60}")
        print(f"Aggregation completed")
        print(f"Files processed: {len(self.files_processed)}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"{'='*60}\n")

        return aggregated_files
