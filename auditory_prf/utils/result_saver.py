"""
Save experimental results in multiple formats (pickle, mat, npz).

Created: 2025-12-02
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np

logger = logging.getLogger(__name__)


class ResultSaver:
    """Save results in multiple file formats with automatic type conversion.

    Supports:
        - Pickle (.pkl): Python-native serialization
        - MATLAB (.mat): For MATLAB compatibility
        - NumPy (.npz): Compressed numpy arrays

    Attributes:
        save_dir: Directory where results will be saved.
    """

    def __init__(self, save_dir: Union[str, Path]):
        """Initialize ResultSaver.

        Args:
            save_dir: Directory path for saving results.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_pickle(
        self,
        data: Any,
        filename: str = "results.pkl"
    ) -> Path:
        """Save data as pickle file.

        Args:
            data: Any Python object to serialize.
            filename: Output filename (default: "results.pkl").

        Returns:
            Path to saved file.
        """
        filepath = self.save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved pickle file: {filepath}")
        return filepath

    def save_mat(
        self,
        data: Dict,
        filename: str = "results.mat",
        oned_as: str = 'column'
    ) -> Path:
        """Save data as MATLAB .mat file.

        Args:
            data: Dictionary containing data to save.
            filename: Output filename (default: "results.mat").
            oned_as: How to save 1D arrays ('column' or 'row').

        Returns:
            Path to saved file.

        Raises:
            ImportError: If scipy is not installed.
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError(
                "scipy is required to save .mat files. "
                "Install it via 'pip install scipy'."
            )

        filepath = self.save_dir / filename
        savemat(filepath, data, oned_as=oned_as)
        logger.info(f"Saved MATLAB file: {filepath}")
        return filepath

    def save_npz(
        self,
        data: Dict,
        filename: str = "results.npz",
        compressed: bool = True
    ) -> Path:
        """Save data as NumPy .npz file.

        Args:
            data: Dictionary containing numpy arrays.
            filename: Output filename (default: "results.npz").
            compressed: If True, use compression (default: True).

        Returns:
            Path to saved file.
        """
        filepath = self.save_dir / filename
        if compressed:
            np.savez_compressed(filepath, **data)
        else:
            np.savez(filepath, **data)
        logger.info(f"Saved NumPy file: {filepath}")
        return filepath

    def save_all(
        self,
        data: Dict,
        base_filename: str = "results",
        formats: List[str] = None,
        **kwargs
    ) -> Dict[str, Path]:
        """Save data in multiple formats.

        Args:
            data: Dictionary containing results.
            base_filename: Base name for files (extensions added auto).
            formats: List of formats to save. Options: 'pickle', 'mat', 'npz'.
                     If None, saves all formats.
            **kwargs: Additional arguments passed to individual save methods.

        Returns:
            Dictionary mapping format names to file paths.

        Example:
            >>> saver = ResultSaver('./output')
            >>> paths = saver.save_all(
            ...     results_dict,
            ...     base_filename='experiment_001',
            ...     formats=['pickle', 'mat']
            ... )
            >>> print(paths['pickle'])
            PosixPath('./output/experiment_001.pkl')
        """
        if formats is None:
            formats = ['pickle', 'mat', 'npz']

        saved_paths = {}

        if 'pickle' in formats:
            saved_paths['pickle'] = self.save_pickle(
                data,
                filename=f"{base_filename}.pkl"
            )

        if 'mat' in formats:
            saved_paths['mat'] = self.save_mat(
                data,
                filename=f"{base_filename}.mat",
                oned_as=kwargs.get('oned_as', 'column')
            )

        if 'npz' in formats:
            saved_paths['npz'] = self.save_npz(
                data,
                filename=f"{base_filename}.npz",
                compressed=kwargs.get('compressed', True)
            )

        logger.info(f"Saved {len(formats)} "
                    f"formats: {list(saved_paths.keys())}")
        return saved_paths

    def load_pickle(self, filename: str = "results.pkl") -> Any:
        """Load data from pickle file.

        Args:
            filename: Pickle file to load.

        Returns:
            Deserialized Python object.
        """
        filepath = self.save_dir / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle file: {filepath}")
        return data

    def load_mat(self, filename: str = "results.mat") -> Dict:
        """Load data from MATLAB .mat file.

        Args:
            filename: MAT file to load.

        Returns:
            Dictionary containing loaded data.

        Raises:
            ImportError: If scipy is not installed.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError(
                "scipy is required to load .mat files. "
                "Install it via 'pip install scipy'."
            )

        filepath = self.save_dir / filename
        data = loadmat(filepath)
        logger.info(f"Loaded MATLAB file: {filepath}")
        return data

    def load_npz(self, filename: str = "results.npz") -> Dict:
        """Load data from NumPy .npz file.

        Args:
            filename: NPZ file to load.

        Returns:
            Dictionary containing loaded arrays.
        """
        filepath = self.save_dir / filename
        data = dict(np.load(filepath, allow_pickle=True))
        logger.info(f"Loaded NumPy file: {filepath}")
        return data


# Convenience function for quick saves
def save_results(
    data: Dict,
    save_dir: Union[str, Path],
    filename: str = "results",
    formats: List[str] = None
) -> Dict[str, Path]:
    """Convenience function to save results in multiple formats.

    Args:
        data: Dictionary containing results to save.
        save_dir: Directory where files will be saved.
        filename: Base filename (without extension).
        formats: List of formats ('pickle', 'mat', 'npz').
                 If None, saves all.

    Returns:
        Dictionary mapping format names to saved file paths.

    Example:
        >>> paths = save_results(
        ...     my_results,
        ...     './output',
        ...     filename='experiment_001',
        ...     formats=['pickle', 'mat']
        ... )
    """
    saver = ResultSaver(save_dir)
    return saver.save_all(data, base_filename=filename, formats=formats)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_data = {
        'temporal_means': np.random.rand(10, 20),
        'frequencies': np.linspace(125, 2500, 20),
        'db_levels': np.array([50, 60, 70, 80, 90]),
        'metadata': {
            'experiment': 'test',
            'date': '2025-12-02'
        }
    }

    # Save in all formats
    saver = ResultSaver('./test_output')
    paths = saver.save_all(
        sample_data,
        base_filename='test_results',
        formats=['pickle', 'mat', 'npz']
    )

    print("\nSaved files:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    # Load back
    loaded_pkl = saver.load_pickle('test_results.pkl')
    print(f"\nLoaded pickle keys: {loaded_pkl.keys()}")
