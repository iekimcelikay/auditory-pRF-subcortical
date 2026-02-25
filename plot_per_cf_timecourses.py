"""
plot_per_cf_timecourses.py
--------------------------
Save one high-quality PNG per CF per stimulus for visual inspection of
firing-rate time courses (including silence periods).

Usage
-----
    python plot_per_cf_timecourses.py                        # uses default path below
    python plot_per_cf_timecourses.py --results_dir models_output/dipc_test_240226_02
    python plot_per_cf_timecourses.py --results_dir models_output/dipc_test_240226_02 \\
                                      --out_dir my_figures \\
                                      --dpi 150 \\
                                      --figsize 18 4

All figures are saved under <out_dir>/ (created automatically).
Figures are never displayed – matplotlib is forced into non-interactive Agg
mode to keep memory usage flat.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # must be set before importing pyplot
import matplotlib.pyplot as plt

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.visualization.plot_cochlea_output import plot_timecourse_per_cf


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_BASE_DIR = Path("./models_output/dipc_test_240226_03")
DEFAULT_OUT_DIR = Path("./figures")

def parse_args():
    p = argparse.ArgumentParser(
        description="Save per-CF PSTH time-course figures for every stimulus in a results folder."
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help=(
            "Path to the results folder that contains .npz files, "
            "OR to a parent folder whose most-recently modified sub-folder will be used. "
            f"Defaults to the most recent sub-folder of {DEFAULT_BASE_DIR}."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Directory where figures are saved. "
            "Defaults to ./figures/per_cf_timecourses/."
        ),
    )
    p.add_argument("--dpi",     type=int,   default=150,      help="Figure resolution (default 150).")
    p.add_argument("--figsize", type=float, nargs=2,
                   default=[18, 4], metavar=("W", "H"),
                   help="Figure width and height in inches (default 18 4).")
    return p.parse_args()


def parse_tone_timing(identifier: str) -> Optional[Tuple[float, float]]:
    """Extract (tone_dur_ms, isi_ms) from a filename identifier.

    Expects tokens of the form ``dur<N>ms`` and ``isi<N>ms`` anywhere in the
    identifier string (e.g. ``dipc_sequence03_fc125hz_dur267ms_isi67ms_...``).
    Returns None if either token is not found.
    """
    dur_match = re.search(r"dur(\d+(?:\.\d+)?)ms", identifier, re.IGNORECASE)
    isi_match = re.search(r"isi(\d+(?:\.\d+)?)ms", identifier, re.IGNORECASE)
    if dur_match and isi_match:
        return float(dur_match.group(1)), float(isi_match.group(1))
    return None


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


def main():
    args = parse_args()

    results_dir = resolve_results_dir(args.results_dir)
    print(f"Results directory : {results_dir}")

    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"ERROR: no .npz files found in {results_dir}")
    print(f"Found {len(npz_files)} .npz file(s)\n")

    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory  : {out_dir}\n")

    saver = ResultSaver(results_dir)
    figsize = tuple(args.figsize)

    for i, npz_file in enumerate(npz_files, 1):
        print(f"[{i}/{len(npz_files)}] {npz_file.name}")

        data            = saver.load_npz(npz_file.name)
        population_psth = data["population_rate_psth"]
        time_axis       = data["time_axis"]
        cf_list         = data["cf_list"]
        identifier      = str(data.get("soundfileid", npz_file.stem))

        print(f"    PSTH shape : {population_psth.shape}  |  duration : {time_axis[-1]:.2f} s")

        tone_markers = parse_tone_timing(identifier)
        if tone_markers:
            print(f"    Tone timing: dur={tone_markers[0]:.0f} ms, isi={tone_markers[1]:.0f} ms")
        else:
            print(f"    Tone timing: not found in identifier – no red markers")

        figs = plot_timecourse_per_cf(
            time_axis       = time_axis,
            population_psth = population_psth,
            cf_list         = cf_list,
            identifier      = identifier,
            save_dir        = out_dir,
            dpi             = args.dpi,
            figsize         = figsize,
            tone_markers    = tone_markers,
        )

        # Close every figure immediately – nothing is displayed
        for fig in figs:
            plt.close(fig)

    total = len(npz_files) * len(cf_list)
    print(f"\nDone. {total} figure(s) saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()


## USAGE
# use the default path (most recent sub-folder of models_output/dipc_test_240226_02)
#python plot_per_cf_timecourses.py

# point at a specific results folder
#python plot_per_cf_timecourses.py --results_dir models_output/dipc_test_240226_02

# customise output location, resolution, and figure size
#python plot_per_cf_timecourses.py \
#    --results_dir models_output/dipc_test_240226_02 \
#    --out_dir figures/per_cf \
#    --dpi 150 \
#    --figsize 22 4