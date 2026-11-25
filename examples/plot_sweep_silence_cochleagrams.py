"""Cochleagrams of the tone/ISI sweep set (family P2), saved as PNGs; safe on a cluster without a display.

Input: the combined cochlea outputs of examples/combine_cf_batches.py, i.e. models_output/<wav folder name>/combined/
*.npz; --run-dir may be the <wav folder name> folder or the combined/ folder itself (keys: soundfileid, metadata, cf_list (n_cf,) Hz, time_axis (n_bins,) s, population_rate_psth (n_cf, n_bins)).

Output (default <run-dir>/cochleagrams/):
  - one PNG per trial: CF x time image of the first --window-s seconds, grey bands = ISIs
  - overview_all.png: all trials of the set, rate across the columns, tone x ISI down the rows, one shared colour scale
  - mean_rate_per_cf.png: mean rate per CF during the tones (trial mean / duty cycle), one line per sweep rate,
    one panel per tone x ISI; the quickest look at whether the drive per channel changes with rate
  - cochleagram_summary.csv: per trial, the mean rate during tones averaged over CFs, and the same restricted to
    the 125-800 Hz CFs (the range the fundamental sweeps) and to the CFs above 1000 Hz

Usage (from the project root)
-----------------------------
    python examples/plot_sweep_silence_cochleagrams.py --run-dir models_output/sweeps_silence_P2_armA_harmonic_cal60_20261125
    python examples/plot_sweep_silence_cochleagrams.py --run-dir <dir> --window-s 0.7 --smooth-ms 3
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

UNIT_S = 20.0 / 120.0               # tone lengths are k x T, ISIs are j x T/2 (family P2)
HALF_UNIT_S = UNIT_S / 2.0
COLORMAP = "Blues"
RATE_COLORS = ("#9ecae1", "#3182bd", "#08306b")         # light -> dark = slow -> fast (single-hue ramp)
CF_TICKS_HZ = (125, 250, 500, 1000, 2000)
HIGH_CF_HZ = 1000.0
FUNDAMENTAL_RANGE_HZ = (125.0, 800.0)


def resolve_combined_dir(run_dir: Path) -> Path:
    """The folder with the combined .npz files: <run_dir>/combined if it exists, else run_dir itself."""
    return run_dir / "combined" if (run_dir / "combined").is_dir() else run_dir


def find_combined_files(run_dir: Path) -> list:
    """Combined .npz files of the tone/ISI trials (silence excluded), sorted by name."""
    return sorted(p for p in resolve_combined_dir(run_dir).glob("*.npz") if "silence" not in p.name)


def load_combined(npz_path: Path) -> dict:
    """Load one combined result.

    Returns
    -------
    dict: soundfileid (str), metadata (dict), cf_list (n_cf,), time_axis (n_bins,), rate (n_cf, n_bins) spikes/s
    """
    with np.load(npz_path, allow_pickle=True) as data:
        return {"soundfileid": str(data["soundfileid"]), "metadata": data["metadata"].item(),
                "cf_list": np.asarray(data["cf_list"], dtype=float), "time_axis": np.asarray(data["time_axis"], dtype=float),
                "rate": np.asarray(data["population_rate_psth"], dtype=float)}


def exact_tone_isi_s(metadata: dict) -> tuple:
    """(tone_s, isi_s) from the name-rounded milliseconds in the metadata (tone = k x T, ISI = j x T/2)."""
    tone_s = max(round(metadata["tone_ms"] / 1000.0 / UNIT_S), 1) * UNIT_S
    isi_s = max(round(metadata["isi_ms"] / 1000.0 / HALF_UNIT_S), 1) * HALF_UNIT_S
    return tone_s, isi_s


def smooth_in_time(rate: np.ndarray, bin_s: float, smooth_ms: float) -> np.ndarray:
    """Moving average along time. rate: (n_cf, n_bins) -> same shape; smooth_ms <= bin width: unchanged."""
    n_smooth = int(round(smooth_ms / 1000.0 / bin_s))
    if n_smooth <= 1:
        return rate
    kernel = np.ones(n_smooth) / n_smooth
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, rate)


def mean_rate_during_tones(result: dict) -> np.ndarray:
    """Per-CF mean rate over the trial divided by the duty cycle tone / (tone + ISI); shape (n_cf,)."""
    tone_s, isi_s = exact_tone_isi_s(result["metadata"])
    return result["rate"].mean(axis=1) / (tone_s / (tone_s + isi_s))


def _cf_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(CF_TICKS_HZ))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def draw_cochleagram(ax, result: dict, window_s: float, smooth_ms: float, vmax: float):
    """CF x time image of the first window_s seconds with the ISIs shaded; returns the mesh."""
    time_axis = result["time_axis"]
    bin_s = float(np.median(np.diff(time_axis)))
    rate = smooth_in_time(result["rate"], bin_s, smooth_ms)
    keep = time_axis <= window_s
    mesh = ax.pcolormesh(time_axis[keep], result["cf_list"], rate[:, keep], vmin=0.0, vmax=vmax,
                         cmap=COLORMAP, shading="nearest", rasterized=True)
    tone_s, isi_s = exact_tone_isi_s(result["metadata"])
    for k in range(int(window_s / (tone_s + isi_s)) + 1):
        ax.axvspan(k * (tone_s + isi_s) + tone_s, min((k + 1) * (tone_s + isi_s), window_s),
                   color="0.7", alpha=0.25, lw=0)
    ax.set_xlim(0, window_s)
    _cf_axis(ax)
    return mesh


def global_vmax(results: list, window_s: float, smooth_ms: float) -> float:
    """99.5th percentile of the smoothed rate in the plotted windows over all trials (one scale for the whole set)."""
    values = []
    for result in results:
        bin_s = float(np.median(np.diff(result["time_axis"])))
        rate = smooth_in_time(result["rate"], bin_s, smooth_ms)
        values.append(rate[:, result["time_axis"] <= window_s].ravel())
    return float(np.percentile(np.concatenate(values), 99.5))


def condition_label(metadata: dict) -> str:
    return (f"{metadata['rate_hz']:g} Hz | tone {metadata['tone_ms']:g} ms | ISI {metadata['isi_ms']:g} ms")


def plot_single(result: dict, out_path: Path, window_s: float, smooth_ms: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    mesh = draw_cochleagram(ax, result, window_s, smooth_ms, vmax)
    ax.set_title(f"{result['metadata'].get('arm', '')} {result['metadata'].get('carrier', '')} | "
                 f"{condition_label(result['metadata'])}", loc="left", fontsize=10)
    ax.set_xlabel("Time (s)  [grey = ISI]")
    ax.set_ylabel("CF (Hz)")
    fig.colorbar(mesh, ax=ax, label="population rate (spikes/s)", pad=0.01)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _grid_axes(results: list) -> tuple:
    rates = sorted({r["metadata"]["rate_hz"] for r in results})
    cells = sorted({(r["metadata"]["tone_ms"], r["metadata"]["isi_ms"]) for r in results})
    return rates, cells


def plot_overview(results: list, out_path: Path, window_s: float, smooth_ms: float, vmax: float) -> None:
    rates, cells = _grid_axes(results)
    fig, axes = plt.subplots(len(cells), len(rates), figsize=(4.3 * len(rates), 2.0 * len(cells) + 1.2),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    mesh = None
    for result in results:
        meta = result["metadata"]
        ax = axes[cells.index((meta["tone_ms"], meta["isi_ms"])), rates.index(meta["rate_hz"])]
        mesh = draw_cochleagram(ax, result, window_s, smooth_ms, vmax)
        ax.set_title(f"tone {meta['tone_ms']:g} / ISI {meta['isi_ms']:g} ms", loc="left", fontsize=8)
    for col, rate_hz in enumerate(rates):
        axes[0, col].annotate(f"{rate_hz:g} Hz", xy=(0.5, 1.3), xycoords="axes fraction", ha="center",
                              fontsize=12, fontweight="bold")
    for col in range(len(rates)):
        axes[-1, col].set_xlabel("Time (s)", fontsize=8)
    for row in range(len(cells)):
        axes[row, 0].set_ylabel("CF (Hz)", fontsize=8)
    assert mesh is not None, "no trials to plot"
    fig.colorbar(mesh, ax=axes, label="population rate (spikes/s), one scale for all panels", shrink=0.4, pad=0.01)
    fig.suptitle(f"Cochleagrams, first {window_s:g} s of each trial (grey = ISI)", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_mean_rate_per_cf(results: list, out_path: Path) -> None:
    rates, cells = _grid_axes(results)
    tones = sorted({tone for tone, _ in cells})
    isis = sorted({isi for _, isi in cells})
    fig, axes = plt.subplots(len(tones), len(isis), figsize=(3.6 * len(isis), 2.6 * len(tones) + 0.8),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for result in results:
        meta = result["metadata"]
        ax = axes[tones.index(meta["tone_ms"]), isis.index(meta["isi_ms"])]
        ax.plot(result["cf_list"], mean_rate_during_tones(result), color=RATE_COLORS[rates.index(meta["rate_hz"]) % 3],
                lw=1.6, label=f"{meta['rate_hz']:g} Hz")
    for row, tone_ms in enumerate(tones):
        for col, isi_ms in enumerate(isis):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(CF_TICKS_HZ))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_title(f"tone {tone_ms:g} / ISI {isi_ms:g} ms", loc="left", fontsize=8)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(title="sweep rate", fontsize=8, title_fontsize=8)
    for col in range(len(isis)):
        axes[-1, col].set_xlabel("CF (Hz)")
    for row in range(len(tones)):
        axes[row, 0].set_ylabel("mean rate during tones (spikes/s)")
    fig.suptitle("Drive per channel by sweep rate (trial mean / duty cycle)", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_summary_csv(results: list, out_path: Path) -> None:
    low_hz, high_hz = FUNDAMENTAL_RANGE_HZ
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["soundfileid", "rate_hz", "tone_ms", "isi_ms", "mean_rate_during_tones_all_cf",
                         "mean_rate_during_tones_cf_125_800", f"mean_rate_during_tones_cf_above_{HIGH_CF_HZ:g}"])
        for result in results:
            meta, cf_hz = result["metadata"], result["cf_list"]
            per_cf = mean_rate_during_tones(result)
            in_range = (cf_hz >= low_hz) & (cf_hz <= high_hz)
            writer.writerow([result["soundfileid"], meta["rate_hz"], meta["tone_ms"], meta["isi_ms"],
                             f"{per_cf.mean():.3f}", f"{per_cf[in_range].mean():.3f}" if in_range.any() else "",
                             f"{per_cf[cf_hz > HIGH_CF_HZ].mean():.3f}" if (cf_hz > HIGH_CF_HZ).any() else ""])


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run-dir", type=str, required=True,
                        help="models_output/<wav folder name> (its combined/ is used), or the combined/ folder itself.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Default: cochleagrams/ next to combined/ (i.e. <wav folder>/cochleagrams).")
    parser.add_argument("--window-s", type=float, default=1.5, help="Seconds of each trial shown (default 1.5).")
    parser.add_argument("--smooth-ms", type=float, default=2.0, help="Moving-average width along time (default 2 ms).")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    combined_dir = resolve_combined_dir(run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else combined_dir.parent / "cochleagrams"
    files = find_combined_files(run_dir)
    assert files, f"no combined .npz files in {combined_dir} (run examples/combine_cf_batches.py first)"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [load_combined(path) for path in files]
    vmax = global_vmax(results, args.window_s, args.smooth_ms)
    for result in results:
        plot_single(result, out_dir / f"{result['soundfileid']}.png", args.window_s, args.smooth_ms, vmax)
    plot_overview(results, out_dir / "overview_all.png", args.window_s, args.smooth_ms, vmax)
    plot_mean_rate_per_cf(results, out_dir / "mean_rate_per_cf.png")
    write_summary_csv(results, out_dir / "cochleagram_summary.csv")
    print(f"{len(results)} trials -> {out_dir} (colour scale max {vmax:.0f} spikes/s)")


if __name__ == "__main__":
    main()
