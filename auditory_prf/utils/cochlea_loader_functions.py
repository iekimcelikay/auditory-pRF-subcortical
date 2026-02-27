# imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, Tuple


# Project imports
from auditory_prf.utils.misc_functions import find_latest_results_folder
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.calculate_population_rate import calculate_population_rate

#  Define paths

def load_cochlea_results(input_dir, weights=None):
    """Load cochlea simulation results and compute population responses.
    Usage:

    results, population_results, cf_list = load_cochlea_results(input_dir)

    Args:
        input_dir: Path to directory containing cochlea results
        weights: Dict with 'hsr', 'msr', 'lsr' weights. If None, uses defaults.

    Returns:
        results: Dict[freq][db] -> raw data from .npz files
        population_results: Dict[freq][db] -> population mean rates array (num_cf,)
        cf_list: Array of CF values
    """
    # Default weights
    if weights is None:
        weights = {
            'hsr': 0.65,
            'msr': 0.28,
            'lsr': 0.12
        }

    # Find latest results folder
    AN_firingrates_dir = find_latest_results_folder(input_dir)
    results_folder = AN_firingrates_dir
    saver = ResultSaver(results_folder)

    # Find all .npz files
    npz_files = list(results_folder.glob('*.npz'))
    print(f"Found {len(npz_files)} .npz files")

    results = {}

    # Load all data files
    for npz_file in npz_files:
        # Parse filename to extract freq and db
        # Format: test011_freq_125.0hz_db_60.npz
        parts = npz_file.stem.split('_')
        freq_idx = parts.index('freq') + 1
        db_idx = parts.index('db') + 1

        freq = float(parts[freq_idx].replace('hz', ''))
        db = float(parts[db_idx])

        # Load data
        data = saver.load_npz(npz_file.name)

        # Store with freq as key
        if freq not in results:
            results[freq] = {}
        results[freq][db] = data

    print(f"Loaded {len(results)} frequencies")

    # Extract CF list from first file
    first_freq = list(results.keys())[0]
    first_db = list(results[first_freq].keys())[0]
    cf_list = results[first_freq][first_db]['cf_list']

    # Compute population responses
    population_results = {}

    for freq, db_dict in results.items():
        population_results[freq] = {}

        for db, data in db_dict.items():
            # Extract mean rates (it's a dictionary wrapped in numpy array)
            mean_rates = data['mean_rates'].item()
            population_mean = calculate_population_rate(mean_rates)
            population_results[freq][db] = population_mean

    return results, population_results, cf_list

def organize_for_eachtone_allCFs(population_results, cf_list, target_db):
    """Organize population results into a 2D spectrogram matrix for a specific dB level.

    Args:
        population_results: Dict[freq][db] -> array of firing rates (num_cf,)
        cf_list: Array of CF values (num_cf,)
        target_db: dB level to extract

    Returns:
        response_matrix: 2D array of shape (num_cf, num_tones)
                        Rows = CF channels, Columns = tone frequencies
        tone_freqs: Sorted array of tone frequencies used as stimuli
    """
    # Get all tone frequencies that have the target dB level
    tone_freqs = []
    for freq, db_dict in population_results.items():
        if target_db in db_dict:
            tone_freqs.append(freq)

    # Sort tone frequencies
    tone_freqs = np.array(sorted(tone_freqs))

    # Initialize response matrix: (num_cf, num_tones)
    num_cf = len(cf_list)
    num_tones = len(tone_freqs)
    response_matrix = np.zeros((num_cf, num_tones))

    # Fill the matrix
    for i_tone, freq in enumerate(tone_freqs):
        response_matrix[:, i_tone] = population_results[freq][target_db]

    print(f"Organized matrix shape: {response_matrix.shape}")
    print(f"  - {num_cf} CF channels")
    print(f"  - {num_tones} tone frequencies")
    print(f"  - dB level: {target_db}")

    return response_matrix, tone_freqs


def resolve_results_dir(path: Optional[Path]) -> Path:
    """Return the directory that actually contains .npz files."""
    if path is None:
        path = DEFAULT_BASE_DIR

    path = path.expanduser().resolve()
    if not path.exists():
        sys.exit(f"ERROR: path does not exist: {path}")

    # If the path itself contains .npz files, use it directly
    if list(path.glob("*.npz")):
        return path

    # Otherwise descend into the most-recently modified sub-directory
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        sys.exit(f"ERROR: no sub-directories found in {path}")
    return max(subdirs, key=lambda d: d.stat().st_mtime)

# Test:
# test passed. functions are working when alone in this script. 28.01.2026

def main():

    input_dir = Path(__file__).parent.parent / "models_output" / "cochlea_test015_approximate"

    results, population_results, cf_list = load_cochlea_results(input_dir)
    response_matrix, tone_freqs = organize_for_eachtone_allCFs(population_results, cf_list, target_db=60)



if __name__ == '__main__':
    main()