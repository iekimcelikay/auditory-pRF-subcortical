"""Cochleagrams of CochleaWavSimulation results for the centered-sweep stimuli.

Reads the .npz files saved by CochleaWavSimulation (population_rate_psth, time_axis,
cf_list, soundfileid) and plots, per file:
  top    : zoom cochleagram (first ZOOM_S seconds)
  middle : cochleagram of the first SEQUENCE_S seconds (zoom window shaded)
  bottom : population rate of the CF channel nearest the sweep center, same 10 s
All files in a folder share one color scale so conditions can be compared directly.

Run from the project root:
    python visualization/scratch/plot_sweep_cochleagrams.py \
        --results-dir models_output/<folder> --out-dir <plot_folder>
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

ZOOM_S = 2.0
SEQUENCE_S = 10.0


def load_result(npz_path: Path) -> dict:
    """Load one CochleaWavSimulation .npz.

    Returns
    -------
    dict with population_rate_psth (n_cf, n_bins), time_axis (n_bins,),
    cf_list (n_cf,), soundfileid (str)
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {key: data[key] for key in ("population_rate_psth", "time_axis", "cf_list")}
    result["soundfileid"] = str(data["soundfileid"])
    n_cf, n_bins = result["population_rate_psth"].shape
    assert result["cf_list"].shape == (n_cf,), "cf_list does not match PSTH rows"
    assert result["time_axis"].shape == (n_bins,), "time_axis does not match PSTH columns"
    return result


def parse_sweep_params(identifier: str) -> dict:
    """center_hz, shift_oct, n_cycles from a centered-sweep seq_id (None if absent)."""
    center = re.search(r"fc(\d+(?:\.\d+)?)hz", identifier)
    shift = re.search(r"shift(\d+(?:\.\d+)?)oct", identifier)
    cycles = re.search(r"_(\d+)cyc", identifier)
    return {"center_hz": float(center.group(1)) if center else None,
            "shift_oct": float(shift.group(1)) if shift else None,
            "n_cycles": int(cycles.group(1)) if cycles else None}


def draw_cochleagram(ax, time_s: np.ndarray, cf_hz: np.ndarray, rate_sps: np.ndarray,
                     t_end_s: float, vmin: float, vmax: float,
                     center_hz: Optional[float] = None):
    """Population rate (n_cf, n_bins) as CF (log) x time image; returns the mesh."""
    keep = time_s <= t_end_s
    mesh = ax.pcolormesh(time_s[keep], cf_hz, rate_sps[:, keep], shading="nearest",
                         cmap="magma", vmin=vmin, vmax=vmax, rasterized=True)
    if center_hz is not None:
        ax.axhline(center_hz, color="white", lw=0.8, ls="--")
    ax.set_yscale("log")
    ticks_hz = [t for t in (125, 250, 500, 1000, 2000) if cf_hz.min() <= t <= cf_hz.max()]
    ax.yaxis.set_major_locator(FixedLocator(ticks_hz))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks_hz]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, t_end_s)
    ax.set_ylabel("CF (Hz, log)")
    return mesh


def plot_result(result: dict, out_path: Path, vmin: float, vmax: float) -> None:
    """Zoom cochleagram, 10 s cochleagram and nearest-CF timecourse for one file."""
    time_s, cf_hz = result["time_axis"], result["cf_list"]
    rate_sps = result["population_rate_psth"]
    params = parse_sweep_params(result["soundfileid"])
    center_hz = params["center_hz"]

    fig, (ax_zoom, ax_seq, ax_cf) = plt.subplots(
        3, 1, figsize=(11, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 0.6]})
    mesh = draw_cochleagram(ax_zoom, time_s, cf_hz, rate_sps, ZOOM_S, vmin, vmax, center_hz)
    draw_cochleagram(ax_seq, time_s, cf_hz, rate_sps, SEQUENCE_S, vmin, vmax, center_hz)
    ax_seq.axvspan(0, ZOOM_S, color="white", alpha=0.18, lw=0)
    ax_zoom.set_title(f"zoom: first {ZOOM_S:.0f} s", fontsize=10, loc="left")
    ax_seq.set_title(f"cochleagram: first {SEQUENCE_S:.0f} s (shaded = zoom window)",
                     fontsize=10, loc="left")
    fig.colorbar(mesh, ax=[ax_zoom, ax_seq], label="population rate (spikes/s)", pad=0.01)

    nearest_idx = int(np.argmin(np.abs(cf_hz - (center_hz if center_hz else cf_hz.mean()))))
    keep = time_s <= SEQUENCE_S
    ax_cf.plot(time_s[keep], rate_sps[nearest_idx, keep], color="tab:blue", lw=0.8)
    ax_cf.set_xlim(0, SEQUENCE_S)
    ax_cf.set_xlabel("time (s)")
    ax_cf.set_ylabel("rate (spikes/s)")
    ax_cf.set_title(f"CF channel nearest the center: {cf_hz[nearest_idx]:.0f} Hz",
                    fontsize=10, loc="left")
    ax_cf.grid(alpha=0.3)

    if None not in params.values():
        fig.suptitle(f"center {center_hz} Hz  |  shift {params['shift_oct']:.2f} oct  |  "
                     f"{params['n_cycles']} cycles", fontsize=11)
    else:
        fig.suptitle(result["soundfileid"], fontsize=11)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    npz_paths = sorted(Path(args.results_dir).glob("*.npz"))
    assert npz_paths, f"no .npz files in {args.results_dir}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [load_result(path) for path in npz_paths]
    vmin = min(result["population_rate_psth"].min() for result in results)
    vmax = max(result["population_rate_psth"].max() for result in results)
    print(f"{len(results)} files, shared color scale {vmin:.1f} ... {vmax:.1f} spikes/s")
    for path, result in zip(npz_paths, results):
        plot_result(result, out_dir / f"{path.stem}.png", vmin, vmax)
        print("wrote", out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
