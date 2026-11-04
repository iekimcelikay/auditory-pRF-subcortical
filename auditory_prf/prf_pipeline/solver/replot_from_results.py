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
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.replot_from_results <run_dir> [<run_dir> ...] [--out_dir DIR] [--workers N]

<run_dir> is any one of the per-timestamp result directories this project's
recovery scripts create, e.g.:
    kim2024_style_duration_recovery_torch_order_debug_results/20261014_164720
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
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
DEFAULT_OUT_DIR = "visualization/scratch/recovery_replots"
DEFAULT_N_WORKERS = 4   # each worker holds ~1M-point figures in memory; keep modest
NOISE_LEVEL_ORDER = ["low", "mid", "high"]

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


def replot_group(out_dir: Path, prefix: str, design_seed: str, timestamp: str, level_paths: dict) -> None:
    """Write every plot/table for one design group into ``out_dir`` (never into the results folder)."""
    tag = f"design{design_seed}_{timestamp}"
    print(f"\n=== {prefix}, design_seed={design_seed}, timestamp={timestamp} "
          f"(levels: {sorted(level_paths)}) ===")

    true_params = None
    param_names = None
    recovered_by_noise = {}
    # Canonical low-mid-high order (glob order is alphabetical: high, low, mid), so every
    # downstream plot (bar chart groups, legends) lists the noise levels the same way.
    for level, path in sorted(level_paths.items(), key=lambda item: NOISE_LEVEL_ORDER.index(item[0])):
        d = np.load(path, allow_pickle=True)
        true_params = d["true_params"]
        recovered_by_noise[level] = d["recovered_params"]
        param_names = list(d["param_names"])

        scatter_path = out_dir / f"{prefix}_scatter_design{design_seed}_{level}_{timestamp}.png"
        table_path   = out_dir / f"{prefix}_table_design{design_seed}_{level}_{timestamp}.csv"
        plot_recovery(true_params, recovered_by_noise[level], scatter_path, param_names=param_names)
        save_recovery_table(true_params, recovered_by_noise[level], table_path, param_names=param_names)

    if len(recovered_by_noise) > 1:
        noise_comparison_path = out_dir / f"{prefix}_noise_comparison_{tag}.png"
        plot_noise_comparison(true_params, recovered_by_noise, noise_comparison_path, param_names=param_names)

        accuracy_by_noise = {
            level: compute_accuracy_correlations(true_params, recovered, param_names=param_names)
            for level, recovered in recovered_by_noise.items()
        }
        n_voxels  = true_params.shape[0]
        n_repeats = next(iter(recovered_by_noise.values())).shape[1]
        accuracy_table_path = out_dir / f"{prefix}_accuracy_table_{tag}.csv"
        accuracy_plot_path  = out_dir / f"{prefix}_accuracy_comparison_{tag}.png"
        save_accuracy_table(accuracy_by_noise, accuracy_table_path, param_names=param_names)
        plot_accuracy_comparison(accuracy_by_noise, accuracy_plot_path, param_names=param_names,
                                  n_voxels=n_voxels, n_repeats=n_repeats)


def _replot_group_task(task: tuple) -> None:
    """Top-level (picklable) wrapper so worker processes can run ``replot_group``."""
    replot_group(*task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate recovery plots/tables from saved results.")
    parser.add_argument("run_dirs", nargs="+", help="one or more per-timestamp result directories")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="where plots/tables are written (default: %(default)s/<results folder name>/); "
                             "the results folders themselves are never written to")
    parser.add_argument("--workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"parallel worker processes, one design group per task (default {DEFAULT_N_WORKERS})")
    args = parser.parse_args()

    tasks = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {run_dir}")
        groups = find_result_groups(run_dir)
        if not groups:
            print(f"No '*_results_design*_{{low,mid,high}}_*.npz' files found in {run_dir}")
        out_dir = Path(args.out_dir) / run_dir.resolve().parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for (prefix, design_seed, timestamp), level_paths in groups.items():
            tasks.append((out_dir, prefix, design_seed, timestamp, level_paths))

    n_workers = max(1, min(args.workers, len(tasks)))
    print(f"Replotting {len(tasks)} design group(s) from {len(args.run_dirs)} folder(s) with {n_workers} worker(s)")
    if n_workers == 1:
        for task in tasks:
            _replot_group_task(task)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_replot_group_task, tasks))   # list() re-raises any worker exception


if __name__ == "__main__":
    main()
