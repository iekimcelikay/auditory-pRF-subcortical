"""replot_from_results.py
==========================
Regenerate plots/tables from already-saved recovery ``.npz`` results,
without re-running the (expensive) candidate-grid build or Monte Carlo
simulation — useful after a plotting-code fix (e.g. the noise-comparison
z-order fix — see project discussion) or any other post-hoc plot tweak.

Works generically across every recovery script in this project (AdapTrans
or duration, forward_model order or torch order, N_VOXELS/N_NOISE_REPEATS
of any size) because:
  - ``param_names`` is saved inside each ``*_results_*.npz`` file itself
    (every script's ``main()`` does ``np.savez(..., param_names=PARAM_NAMES)``),
    so this script never needs to import a model-specific constant.
  - n_voxels/n_repeats (needed only for the accuracy plot's title) are read
    directly from the loaded arrays' shapes, not from any hardcoded
    constant.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.replot_from_results <run_dir>

<run_dir> is any one of the per-timestamp result directories this project's
recovery scripts create, e.g.:
    kim2024_style_duration_recovery_torch_order_debug_results/20261014_164720
"""

import re
import sys
from pathlib import Path

import numpy as np

from auditory_prf.prf_pipeline.solver.grid_recovery import (
    plot_recovery,
    save_recovery_table,
    plot_noise_comparison,
)
from auditory_prf.prf_pipeline.solver.kim2024_style_recovery import (
    compute_accuracy_correlations,
    save_accuracy_table,
    plot_accuracy_comparison,
)

# Matches e.g. "kim_duration_recovery_results_design1000_low_20261014_164720.npz"
# -> prefix="kim_duration_recovery", design_seed="1000", level="low", timestamp="20261014_164720"
_RESULTS_RE = re.compile(r"^(?P<prefix>.+)_results_design(?P<design_seed>[^_]+)_(?P<level>low|mid|high)_(?P<timestamp>\d{8}_\d{6})\.npz$")


def find_result_groups(run_dir: Path) -> dict:
    """Group every ``*_results_design<N>_<level>_<timestamp>.npz`` file in
    ``run_dir`` by (prefix, design_seed, timestamp), collecting all noise
    levels found for that group together.

    Returns
    -------
    dict
        ``{(prefix, design_seed, timestamp): {level: Path}}``
    """
    groups: dict = {}
    for f in sorted(run_dir.glob("*_results_design*.npz")):
        m = _RESULTS_RE.match(f.name)
        if not m:
            print(f"  (skipping unrecognized file name: {f.name})")
            continue
        key = (m["prefix"], m["design_seed"], m["timestamp"])
        groups.setdefault(key, {})[m["level"]] = f
    return groups


def replot_group(run_dir: Path, prefix: str, design_seed: str, timestamp: str, level_paths: dict) -> None:
    tag = f"design{design_seed}_{timestamp}"
    print(f"\n=== {prefix}, design_seed={design_seed}, timestamp={timestamp} "
          f"(levels: {sorted(level_paths)}) ===")

    true_params = None
    param_names = None
    recovered_by_noise = {}
    for level, path in level_paths.items():
        d = np.load(path, allow_pickle=True)
        true_params = d["true_params"]
        recovered_by_noise[level] = d["recovered_params"]
        param_names = list(d["param_names"])

        scatter_path = run_dir / f"{prefix}_scatter_design{design_seed}_{level}_{timestamp}.png"
        table_path   = run_dir / f"{prefix}_table_design{design_seed}_{level}_{timestamp}.csv"
        plot_recovery(true_params, recovered_by_noise[level], scatter_path, param_names=param_names)
        save_recovery_table(true_params, recovered_by_noise[level], table_path, param_names=param_names)

    if len(recovered_by_noise) > 1:
        noise_comparison_path = run_dir / f"{prefix}_noise_comparison_{tag}.png"
        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_path, param_names=param_names)

        accuracy_by_noise = {
            level: compute_accuracy_correlations(true_params, recovered, param_names=param_names)
            for level, recovered in recovered_by_noise.items()
        }
        n_voxels  = true_params.shape[0]
        n_repeats = next(iter(recovered_by_noise.values())).shape[1]
        accuracy_table_path = run_dir / f"{prefix}_accuracy_table_{tag}.csv"
        accuracy_plot_path  = run_dir / f"{prefix}_accuracy_comparison_{tag}.png"
        save_accuracy_table(accuracy_by_noise, accuracy_table_path, param_names=param_names)
        plot_accuracy_comparison(accuracy_by_noise, accuracy_plot_path, param_names=param_names,
                                  n_voxels=n_voxels, n_repeats=n_repeats)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python -m auditory_prf.prf_pipeline.solver.replot_from_results <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    groups = find_result_groups(run_dir)
    if not groups:
        print(f"No '*_results_design*_{{low,mid,high}}_*.npz' files found in {run_dir}")
        return

    for (prefix, design_seed, timestamp), level_paths in groups.items():
        replot_group(run_dir, prefix, design_seed, timestamp, level_paths)


if __name__ == "__main__":
    main()
