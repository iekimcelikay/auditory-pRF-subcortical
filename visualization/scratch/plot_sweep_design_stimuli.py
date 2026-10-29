"""Plot every stimulus of sweep_design_centered.py: zoom (top) + first 10 s (bottom).

Each figure is a spectrogram of the rendered audio with the target f(t) dashed. The
y-axis is the 0.55 oct band of each center for BOTH shifts, so the 0.01 oct (inaudible)
trials are drawn on the same scale and look, correctly, almost flat.

Usage
-----
    python visualization/scratch/plot_sweep_design_stimuli.py --out-dir /path/to/plots
    python visualization/scratch/plot_sweep_design_stimuli.py --out-dir /path/to/plots \
        --shifts 0.01
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).parents[2]))

import stimuli.sweep_design_centered as sweep_design

ZOOM_S = 2.0
SEQUENCE_S = 10.0
DB_RANGE = 70.0       # color range below the maximum, in dB


def draw_panel(ax, trial_sound: np.ndarray, center_hz: float, period_s: float,
               shift_oct: float, t_end_s: float, nperseg: int, noverlap: int) -> None:
    """Spectrogram of trial_sound[:t_end_s] with the target f(t) dashed."""
    band_low_hz, band_high_hz = sweep_design.sweep_endpoints_hz(
        center_hz, sweep_design.AUDIBLE_SHIFT_OCT)
    n_samples = int(t_end_s * sweep_design.SAMPLE_RATE_HZ)
    freq_hz, time_s, power = spectrogram(trial_sound[:n_samples],
                                         sweep_design.SAMPLE_RATE_HZ,
                                         nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(power + 1e-20)
    ax.pcolormesh(time_s, freq_hz, power_db, shading="auto", cmap="magma",
                  vmin=power_db.max() - DB_RANGE, vmax=power_db.max(), rasterized=True)
    t_fine_s = np.linspace(0, t_end_s, 4000)
    target_fn_hz = sweep_design.triangle_freq_fn_hz(center_hz, period_s, shift_oct)
    ax.plot(t_fine_s, target_fn_hz(t_fine_s), color="white", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(band_low_hz * 0.92, band_high_hz * 1.08)
    ticks_hz = [band_low_hz, center_hz, band_high_hz]
    ax.yaxis.set_major_locator(FixedLocator(ticks_hz))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{tick:.0f}" for tick in ticks_hz]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, t_end_s)
    ax.set_ylabel("frequency (Hz, log)")


def plot_stimulus(center_hz: float, n_cycles: int, shift_oct: float, out_path: Path) -> None:
    """Zoom panel above, first-10-s panel below, for one stimulus."""
    trial_sound = sweep_design.render_centered_sweep(center_hz, n_cycles, shift_oct)
    period_s = sweep_design.SWEEP_DURATION_S / n_cycles
    low_hz, high_hz = sweep_design.sweep_endpoints_hz(center_hz, shift_oct)
    fig, (ax_zoom, ax_seq) = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    draw_panel(ax_zoom, trial_sound, center_hz, period_s, shift_oct, ZOOM_S, 2048, 1920)
    draw_panel(ax_seq, trial_sound, center_hz, period_s, shift_oct, SEQUENCE_S, 4096, 3072)
    ax_seq.axvspan(0, ZOOM_S, color="white", alpha=0.18, lw=0)
    ax_zoom.set_title(f"zoom: first {ZOOM_S:.0f} s", fontsize=10, loc="left")
    ax_seq.set_title(f"sequence: first {SEQUENCE_S:.0f} s of "
                     f"{sweep_design.TRIAL_DURATION_S:.0f} s (shaded = zoom window)",
                     fontsize=10, loc="left")
    ax_seq.set_xlabel("time (s)")
    rise_fraction = sweep_design.RISE_FRACTION
    rise_speed, fall_speed = sweep_design.sweep_speeds_oct_per_s(shift_oct, n_cycles)
    fig.suptitle(
        f"center {center_hz} Hz  |  shift {shift_oct:.2f} oct ({low_hz:.1f}-{high_hz:.1f} Hz)"
        f"  |  {n_cycles} cycles ({n_cycles / sweep_design.SWEEP_DURATION_S:.2f} Hz)"
        f"  |  rise {rise_fraction * period_s * 1000:.0f} ms / "
        f"fall {(1 - rise_fraction) * period_s * 1000:.0f} ms"
        f"  |  {rise_speed:.2f} / {fall_speed:.2f} oct/s", fontsize=11)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--shifts", type=float, nargs="+",
                        default=[sweep_design.AUDIBLE_SHIFT_OCT,
                                 sweep_design.INAUDIBLE_SHIFT_OCT],
                        help="Shifts in octaves to plot (default: both design shifts).")
    args = parser.parse_args()
    for shift_oct in args.shifts:
        shift_dir = Path(args.out_dir) / f"shift_{shift_oct:.2f}oct"
        shift_dir.mkdir(parents=True, exist_ok=True)
        for center_hz, n_cycles in itertools.product(sweep_design.CENTER_FREQS_HZ,
                                                     sweep_design.CYCLES_PER_TRIAL):
            seq_id = sweep_design.centered_seq_id(shift_oct, n_cycles, 0, center_hz)
            plot_stimulus(center_hz, n_cycles, shift_oct, shift_dir / f"{seq_id}.png")
        print(f"shift {shift_oct:.2f} oct: {len(list(shift_dir.glob('*.png')))} plots "
              f"in {shift_dir}")


if __name__ == "__main__":
    main()
