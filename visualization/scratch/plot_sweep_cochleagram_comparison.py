"""Side-by-side cochleagrams: audible (0.55 oct) vs inaudible (0.01 oct) per stimulus.

Searches --results-root recursively for CochleaWavSimulation .npz files (each SLURM
array task writes into its own timestamped subfolder), pairs the shifts by
(center, n_cycles), and plots one figure per pair for which BOTH shifts exist. Left
column = larger shift, right column = smaller shift. Per column:
  top    : zoom cochleagram (first ZOOM_S seconds)
  middle : cochleagram of the first SEQUENCE_S seconds (zoom window shaded)
  bottom : population rate of the CF channel nearest the center (shared y-axis)
One color scale is shared across ALL figures and both columns.

Run from the project root:
    python visualization/scratch/plot_sweep_cochleagram_comparison.py \
        --results-root models_output --out-dir <plot_folder>
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import plot_sweep_cochleagrams as cochleagrams

SWEEP_FOLDER_GLOB = "sweep_design_*"


def find_sweep_results(results_root: Path) -> dict:
    """Map (center_hz, n_cycles, shift_oct) -> .npz path for every sweep result found.

    If a stimulus was run more than once, the path that sorts last (newest
    timestamped folder) wins.
    """
    found = {}
    for npz_path in sorted(results_root.glob(f"{SWEEP_FOLDER_GLOB}/**/*.npz")):
        params = cochleagrams.parse_sweep_params(npz_path.stem)
        if None in params.values():
            continue
        found[(params["center_hz"], params["n_cycles"], params["shift_oct"])] = npz_path
    return found


def pair_by_stimulus(found: dict) -> list:
    """List of (center_hz, n_cycles, [shift_oct, ...]) with 2+ shifts, larger shift first."""
    shifts_by_stimulus = {}
    for center_hz, n_cycles, shift_oct in found:
        shifts_by_stimulus.setdefault((center_hz, n_cycles), []).append(shift_oct)
    return [(center_hz, n_cycles, sorted(shifts, reverse=True))
            for (center_hz, n_cycles), shifts in sorted(shifts_by_stimulus.items())
            if len(shifts) >= 2]


def plot_pair(results_by_shift: dict, center_hz: float, n_cycles: int,
              out_path: Path, vmin: float, vmax: float) -> None:
    """One column per shift: zoom cochleagram, 10 s cochleagram, nearest-CF timecourse."""
    shifts = list(results_by_shift)
    fig, axes = plt.subplots(3, len(shifts), figsize=(7.5 * len(shifts), 10),
                             constrained_layout=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.6]})
    mesh = None
    for col, shift_oct in enumerate(shifts):
        result = results_by_shift[shift_oct]
        time_s, cf_hz = result["time_axis"], result["cf_list"]
        rate_sps = result["population_rate_psth"]
        band_low_hz = center_hz * 2 ** (-shift_oct / 2)
        band_high_hz = center_hz * 2 ** (shift_oct / 2)
        ax_zoom, ax_seq, ax_cf = axes[:, col]
        mesh = cochleagrams.draw_cochleagram(ax_zoom, time_s, cf_hz, rate_sps,
                                             cochleagrams.ZOOM_S, vmin, vmax, center_hz)
        cochleagrams.draw_cochleagram(ax_seq, time_s, cf_hz, rate_sps,
                                      cochleagrams.SEQUENCE_S, vmin, vmax, center_hz)
        for ax in (ax_zoom, ax_seq):
            for edge_hz in (band_low_hz, band_high_hz):
                ax.axhline(edge_hz, color="white", lw=0.6, ls=":")
        ax_seq.axvspan(0, cochleagrams.ZOOM_S, color="white", alpha=0.18, lw=0)
        ax_zoom.set_title(f"shift {shift_oct:.2f} oct ({band_low_hz:.0f}-{band_high_hz:.0f} Hz)"
                          f"  |  zoom: first {cochleagrams.ZOOM_S:.0f} s",
                          fontsize=10, loc="left")
        ax_seq.set_title(f"first {cochleagrams.SEQUENCE_S:.0f} s (shaded = zoom window)",
                         fontsize=10, loc="left")
        nearest_idx = int(np.argmin(np.abs(cf_hz - center_hz)))
        keep = time_s <= cochleagrams.SEQUENCE_S
        ax_cf.plot(time_s[keep], rate_sps[nearest_idx, keep], color="tab:blue", lw=0.8)
        ax_cf.set_xlim(0, cochleagrams.SEQUENCE_S)
        ax_cf.set_ylim(vmin, vmax)
        ax_cf.set_xlabel("time (s)")
        ax_cf.set_title(f"CF channel nearest the center: {cf_hz[nearest_idx]:.0f} Hz",
                        fontsize=10, loc="left")
        ax_cf.grid(alpha=0.3)
        if col == 0:
            ax_cf.set_ylabel("rate (spikes/s)")
        else:
            for ax in (ax_zoom, ax_seq, ax_cf):
                ax.set_ylabel("")
    fig.colorbar(mesh, ax=axes[:2, :], label="population rate (spikes/s)", pad=0.01)
    fig.suptitle(f"center {center_hz} Hz  |  {n_cycles} cycles "
                 f"({n_cycles / 19.9:.2f} Hz)  |  white dotted = sweep band edges", fontsize=11)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="models_output",
                        help="Folder containing the sweep_design_* result folders.")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    found = find_sweep_results(Path(args.results_root))
    pairs = pair_by_stimulus(found)
    assert pairs, f"no stimulus has results for 2+ shifts under {args.results_root}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {key: cochleagrams.load_result(found[key])
              for center_hz, n_cycles, shifts in pairs
              for key in ((center_hz, n_cycles, shift_oct) for shift_oct in shifts)}
    vmin = min(result["population_rate_psth"].min() for result in loaded.values())
    vmax = max(result["population_rate_psth"].max() for result in loaded.values())
    print(f"{len(found)} sweep results found, {len(pairs)} comparable stimuli, "
          f"shared color scale {vmin:.1f} ... {vmax:.1f} spikes/s")
    for center_hz, n_cycles, shifts in pairs:
        results_by_shift = {shift_oct: loaded[(center_hz, n_cycles, shift_oct)]
                            for shift_oct in shifts}
        out_path = out_dir / f"compare_fc{center_hz:.1f}hz_{n_cycles:03d}cyc.png"
        plot_pair(results_by_shift, center_hz, n_cycles, out_path, vmin, vmax)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
