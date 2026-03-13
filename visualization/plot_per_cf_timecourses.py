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
import gc
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # must be set before importing pyplot
import matplotlib.pyplot as plt

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.visualization.plot_cochlea_output import plot_timecourse_per_cf
from auditory_prf.utils.cochlea_loader_functions import resolve_results_dir

# ── defaults ────────────────────────────────────────────────────────────────
EXP_NAME = "dipc_test_250225_01"   # ← change this when running from the IDE

DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")
DEFAULT_OUT_DIR  = Path(f"./figures/{EXP_NAME}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Save per-CF PSTH time-course figures for every stimulus in a results folder."
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default=EXP_NAME,
        help=(
            f"Experiment name. Defaults to '{EXP_NAME}' (set at top of script). "
            "Overrides DEFAULT_BASE_DIR and DEFAULT_OUT_DIR when provided."
        ),
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
                   default=[12, 4], metavar=("W", "H"),
                   help="Figure width and height in inches (default 12 4).")
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




def main():
    args = parse_args()

    # Derive paths from exp_name (CLI value overrides the module-level default)
    base_dir        = Path(f"./models_output/{args.exp_name}")
    out_dir_default = Path(f"./figures/{args.exp_name}")

    results_dir = resolve_results_dir(args.results_dir if args.results_dir is not None else base_dir)
    print(f"Experiment        : {args.exp_name}")
    print(f"Results directory : {results_dir}")

    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"ERROR: no .npz files found in {results_dir}")
    print(f"Found {len(npz_files)} .npz file(s)\n")

    out_dir = args.out_dir if args.out_dir is not None else out_dir_default
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

        # Close every figure and free all loaded arrays before the next file
        n_cf = len(cf_list)
        for fig in figs:
            plt.close(fig)
        del data, population_psth, time_axis, cf_list, identifier, figs
        gc.collect()

    total = len(npz_files) * n_cf
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
#    --figsize 22 4/home/ekim/auditory-pRF-subcortical/auditory_prf/visualization/plot_cochlea_output.py:191: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`."Returns to baseline" needs a criterion — you could define it as when the rate drops below baseline + N×SD of the baseline, where N=1 or 2. This handles the noisy 1 ms bins gracefully.