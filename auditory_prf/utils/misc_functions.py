from pathlib import Path


# Find most recent results folders
def find_latest_results_folder(base_dir):
    """Find the most recent results folder in the given directory."""
    results_folders = list(Path(base_dir).glob('results_*'))
    if not results_folders:
        raise FileNotFoundError(f"No results folders found in {base_dir}")
    return max(results_folders, key=lambda p: p.stat().st_mtime)


