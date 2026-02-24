"""
Example script for plotting PSTH time courses from WAV simulation results.

This script loads PSTH data saved from CochleaWavSimulation and visualizes
the population response time courses for each cochlear channel.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.visualization.plot_cochlea_output import plot_psth_timecourses


def load_psth_data(results_dir, filename=None):
    """Load PSTH data from saved results.

    Args:
        results_dir: Path to directory containing saved results
        filename: Specific .npz file to load (without extension).
                  If None, loads the first .npz file found.

    Returns:
        data: Dict containing PSTH data with keys:
            - 'population_rate_psth': array (num_cf, n_bins)
            - 'time_axis': array (n_bins,)
            - 'cf_list': array (num_cf,)
            - 'soundfileid': identifier string
            - 'metadata': dict with file metadata
    """
    results_dir = Path(results_dir)
    saver = ResultSaver(results_dir)

    if filename is None:
        # Find first .npz file
        npz_files = list(results_dir.glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {results_dir}")
        filename = npz_files[0].stem
        print(f"Loading: {filename}")

    # Load data
    data = saver.load_npz(f"{filename}.npz")

    # Verify required fields are present
    required_fields = ['population_rate_psth', 'time_axis', 'cf_list']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return data


def main():
    """Main function demonstrating PSTH time course plotting."""

    # ====================
    # 1. SPECIFY DATA PATH
    # ====================
    # Update this to point to your results directory
    results_dir = Path("./models_output/cochlea_wavfiles_120226_02")

    # Optional: specify a particular file to load
    # Leave as None to load the first file found
    specific_file = None  # e.g., "customsequences_test02_sequence01_fc440hz"

    # ====================
    # 2. LOAD DATA
    # ====================
    print("Loading PSTH data...")
    data = load_psth_data(results_dir, filename=specific_file)

    # Extract required arrays
    population_psth = data['population_rate_psth']
    time_axis = data['time_axis']
    cf_list = data['cf_list']
    identifier = data.get('soundfileid', 'unknown')

    print(f"\nData loaded successfully:")
    print(f"  Sound ID: {identifier}")
    print(f"  PSTH shape: {population_psth.shape}")
    print(f"  Time axis shape: {time_axis.shape}")
    print(f"  Number of CFs: {len(cf_list)}")
    print(f"  Duration: {time_axis[-1]:.2f} seconds")

    # ====================
    # 3. PLOT TIME COURSES
    # ====================
    print("\nPlotting time courses...")

    fig, axes = plot_psth_timecourses(
        time_axis=time_axis,
        population_psth=population_psth,
        cf_list=cf_list,
        identifier=identifier,
        save_path=None  # Set to a path to save figure
    )

    plt.show()

    # ====================
    # 4. OPTIONAL: BATCH PROCESSING
    # ====================
    # To process multiple files, uncomment below:
    """
    print("\n" + "="*60)
    print("Processing all files in directory...")

    npz_files = sorted(results_dir.glob('*.npz'))

    for npz_file in npz_files:
        print(f"\nProcessing: {npz_file.stem}")

        data = load_psth_data(results_dir, filename=npz_file.stem)

        fig, axes = plot_psth_timecourses(
            time_axis=data['time_axis'],
            population_psth=data['population_rate_psth'],
            cf_list=data['cf_list'],
            identifier=data.get('soundfileid', 'unknown'),
            save_path=results_dir / f"psth_timecourse_{npz_file.stem}.png"
        )

        plt.close(fig)  # Close to free memory

    print("\nBatch processing complete!")
    """


if __name__ == '__main__':
    main()
