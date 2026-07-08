"""
visualize_adaptrans_pertone.py
==============================
Visualise the AdapTrans ON and OFF filters applied per-tone for one stimulus
sequence.

For each tone repetition in the sequence, an isolated boxcar train is built
from the pRF response scalar (sharpened & normalised
-to the previous mean across CF- mean AN rate × duration Gaussian),
passed through the AdapTrans ON and OFF filters,
and superposed with a ratio parameter (on_off_ratio) to produce the full
combined timecourse:

    combined = on_off_ratio * on_response + off_response

Layout (N_tones + 2 rows):
  Row 0          : accumulated boxcar train (all tones summed, for reference)
  Rows 1–N_tones : per-tone boxcar (shaded) + ON (solid) + OFF (dashed)
  Row N_tones+1  : superposed ON (orange) + OFF (blue) + combined (purple)

Outputs saved to <out_dir>/:
  <stem>.png  — figure
  <stem>.npz  — all arrays + metadata (loadable with ResultSaver.load_npz)
                includes per_tone_on, per_tone_off, on_response, off_response,
                combined_response, on_off_ratio

Usage
-----
    python visualize_adaptrans_pertone.py
    python visualize_adaptrans_pertone.py --exp_name dipc_test_250225_01 \\
        --seq_index 5 --cf 10 --alpha 2.0 --pref_dur 200 --sigma_dur 20 \\
        --w 0.8 --on_off_ratio 2.0
    python visualize_adaptrans_pertone.py --out_dir figures/adaptrans_check \\
        --on_off_ratio 0.5
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from auditory_prf.utils.cochlea_loader_functions import resolve_results_dir
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_cf_timecourse, load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import (
    build_prf_boxcar_train,
    apply_adaptrans,
)
from auditory_prf.visualization.colormaps import auditory_cortex_cmap

# ── defaults (change here when running from the IDE) ────────────────────────
EXP_NAME         = "dipc_test_250225_01"
DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")
DEFAULT_OUT_DIR  = Path(f"./figures/{EXP_NAME}/adaptrans_pertone")

logger = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise per-tone AdapTrans ON + OFF filters for one stimulus."
    )
    p.add_argument("--exp_name",    type=str,   default=EXP_NAME)
    p.add_argument("--results_dir", type=Path,  default=None,
                   help="Path to the .npz results folder (or parent; latest "
                        "sub-folder is resolved automatically).")
    p.add_argument("--out_dir",     type=Path,  default=None,
                   help=f"Output directory. Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--seq_index",   type=int,   default=15,
                   help="0-based index into the sorted list of .npz files (default 15).")
    p.add_argument("--cf",          type=int,   default=15,
                   help="CF index to extract from the population PSTH (default 15).")
    p.add_argument("--alpha",       type=float, default=8.0,
                   help="Power-law sharpening exponent (default 8.0).")
    p.add_argument("--pref_dur",    type=float, default=200.0,
                   help="Preferred duration for the Gaussian kernel in ms (default 200).")
    p.add_argument("--sigma_dur",   type=float, default=60.0,
                   help="Sigma of the duration Gaussian in ms (default 60).")
    p.add_argument("--w",           type=float, default=0.75,
                   help="AdapTrans adaptation weight (default 0.75).")
    p.add_argument("--K",           type=int,   default=None,
                   help="AdapTrans kernel length in samples (auto if omitted).")
    p.add_argument("--on_off_ratio",   type=float, default=1.0,
                   help="Ratio of ON to OFF gain. >1 onset-dominated, <1 offset-dominated (default 1.0).")
    p.add_argument("--dpi",         type=int,   default=150)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _save_fig(fig, plot_dir: Path, name: str, dpi: int = 150):
    """Save figure to plot_dir and close it."""
    fig.savefig(plot_dir / name, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.debug("  Saved intermediate plot: %s", name)


def _add_tone_markers(ax, onsets_ms, offsets_ms, n_1ms):
    """Shade each tone-on period in light blue."""
    for on, off in zip(onsets_ms, offsets_ms):
        ax.axvspan(on, min(off, n_1ms), alpha=0.10, color="steelblue",
                   linewidth=0, zorder=0)


def _build_output_stem(seq_id, cf_index, cf_hz, alpha, pref_dur, sigma_dur,
                       w, on_off_ratio):
    """Construct an informative base filename encoding all key parameters."""
    return (
        f"adaptrans_pertone_{seq_id}"
        f"_cfidx{cf_index}_cfhz{cf_hz:.0f}"
        f"_alpha{alpha:.1f}_tau{pref_dur:.0f}"
        f"_sigma{sigma_dur:.0f}_w{w:.2f}"
        f"_onoff{on_off_ratio:.2f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    base_dir = Path(f"./models_output/{args.exp_name}")
    out_dir  = (args.out_dir or Path(f"./figures/{args.exp_name}/adaptrans_pertone")).expanduser().resolve()

    # Logging
    LoggingConfigurator(
        output_dir=out_dir,
        log_filename="visualize_adaptrans_pertone.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    ).setup()

    # ── 1. Resolve paths ──────────────────────────────────────────────────────
    results_dir = resolve_results_dir(
        args.results_dir if args.results_dir is not None else base_dir
    )
    logger.info("Experiment        : %s", args.exp_name)
    logger.info("Results directory : %s", results_dir)

    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        logger.error("No .npz files found in %s", results_dir)
        sys.exit(1)
    logger.info("Found %d .npz file(s) | using index %d", len(npz_files), args.seq_index)

    npz_path = npz_files[args.seq_index]
    logger.info("Processing: %s", npz_path.name)

    # ── 2. Load full population PSTH (all CFs × all time bins) ──────────────────────────
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, args.cf)
    dt_s         = time_axis[1] - time_axis[0]
    total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
    n_1ms        = math.ceil(total_dur_ms)
    t_ms         = np.arange(n_1ms, dtype=float)
    logger.info("CF index: %d | CF: %.1f Hz | seq_id: %s", cf_index, cf_hz, seq_id)
    logger.info("total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

    # ── intermediate plot directory ──────────────────────────────────────────
    intermed_dir = out_dir / "intermediate_plots"
    intermed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Intermediate plots → %s", intermed_dir)

    # ── 3. Power-law sharpening across all CFs, preserve grand mean, extract target CF ──
    sharpened_pop = apply_powerlaw_population(population_psth, args.alpha)
    sharpened = sharpened_pop[cf_index, :]
    logger.debug("Sharpened timecourse shape: %s", sharpened.shape)

    # --- Plot: powerlaw sharpening (raw vs sharpened for target CF)
    raw_cf = population_psth[cf_index, :]
    fig, axes_pl = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes_pl[0].plot(time_axis * 1000, raw_cf, linewidth=0.6, color='steelblue')
    axes_pl[0].set_ylabel("Rate (spk/s)")
    axes_pl[0].set_title(f"Raw CF timecourse — CF {cf_hz:.0f} Hz | {seq_id}")
    axes_pl[1].plot(time_axis * 1000, sharpened, linewidth=0.6, color='darkorange')
    axes_pl[1].set_ylabel("Rate (spk/s)")
    axes_pl[1].set_xlabel("Time (ms)")
    axes_pl[1].set_title(f"After power-law sharpening (α={args.alpha})")
    _save_fig(fig, intermed_dir, f"01_powerlaw_{seq_id}.png", dpi=args.dpi)

    # ── 4. Chunk into tone-on windows ─────────────────────────────────────────
    result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id)
    n_tones       = len(result["chunks"])
    mean_rates_on = [np.mean(c) for c in result["chunks"]]
    logger.info("tone_dur: %.1f ms | ISI: %.1f ms | n_tones: %d",
                tone_dur_ms, isi_ms, n_tones)

    # --- Plot: mean firing rate per tone-ON chunk
    tone_indices = np.arange(1, n_tones + 1)
    fig, ax_ch = plt.subplots(figsize=(10, 4))
    ax_ch.bar(tone_indices, mean_rates_on, color='teal', edgecolor='k', linewidth=0.5)
    ax_ch.set_xlabel("Tone number")
    ax_ch.set_ylabel("Mean rate (spk/s)")
    ax_ch.set_title(f"Tone-ON mean rates — {seq_id} | dur={tone_dur_ms:.0f}ms, ISI={isi_ms:.0f}ms")
    _save_fig(fig, intermed_dir, f"02_chunk_mean_rates_{seq_id}.png", dpi=args.dpi)

    # ── 5. Duration Gaussian (scalar) ─────────────────────────────────────────
    prf_responses = [
        apply_duration_gaussian_scalar(m, tone_dur_ms, args.pref_dur, args.sigma_dur)
        for m in mean_rates_on
    ]
    logger.info("pRF responses: %s", [f"{p:.3e}" for p in prf_responses])

    # --- Plot: duration-weighted pRF responses vs raw mean rates
    fig, ax_dw = plt.subplots(figsize=(10, 4))
    ax_dw.bar(tone_indices - 0.2, mean_rates_on, width=0.4,
              label="Raw mean rate", color='teal', alpha=0.6)
    ax_dw.bar(tone_indices + 0.2, prf_responses, width=0.4,
              label=f"Duration-weighted (τ₀={args.pref_dur:.0f}, σ={args.sigma_dur:.0f})",
              color='salmon', alpha=0.8)
    ax_dw.set_xlabel("Tone number")
    ax_dw.set_ylabel("Rate (spk/s)")
    ax_dw.set_title(f"Duration Gaussian weighting — {seq_id}")
    ax_dw.legend()
    _save_fig(fig, intermed_dir, f"03_duration_weighted_{seq_id}.png", dpi=args.dpi)

    # ── 6-7. Per-tone AdapTrans ON + OFF, then superpose ─────────────────────
    per_tone_trains   = []
    per_tone_on       = []
    per_tone_off      = []
    on_response       = np.zeros(n_1ms)
    off_response      = np.zeros(n_1ms)
    train_accumulated = np.zeros(n_1ms)

    for s, (prf_s, on_ms, off_ms) in enumerate(
            zip(prf_responses, result["onsets_ms"], result["offsets_ms"])):

        single_train = build_prf_boxcar_train(
            [prf_s], np.array([on_ms]), np.array([off_ms]),
            total_dur_ms, dt_ms=1.0,
        )
        per_tone_trains.append(single_train.copy())
        train_accumulated += single_train

        on_off_s = apply_adaptrans(
            single_train[np.newaxis, :],
            CFs_Hz=np.array([cf_hz]),
            dt_ms=1.0,
            w=args.w,
            K=args.K,
            pad_value=0.0,
        )
        per_tone_on.append(on_off_s[0, 0, :].copy())
        per_tone_off.append(on_off_s[1, 0, :].copy())
        on_response  += per_tone_on[-1]
        off_response += per_tone_off[-1]
        logger.debug("  Tone %d/%d | ON max: %.4e | OFF max: %.4e",
                     s + 1, n_tones, per_tone_on[-1].max(), per_tone_off[-1].max())

    logger.info("ON  response shape: %s | max: %.4e", on_response.shape, on_response.max())
    logger.info("OFF response shape: %s | max: %.4e", off_response.shape, off_response.max())

    # ── 8. Figure ─────────────────────────────────────────────────────────────
    colors = auditory_cortex_cmap(np.linspace(0, 1, max(n_tones, 3)))
    n_rows = 1 + n_tones + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.4 * n_rows), sharex=True)
    fig.suptitle(
        f"{seq_id}\n"
        f"CF={cf_hz:.1f} Hz (idx {cf_index}) | "
        f"α={args.alpha} | τ₀={args.pref_dur:.0f} ms | σ={args.sigma_dur:.0f} ms | "
        f"w={args.w} | K={'auto' if args.K is None else args.K} | "
        f"on_off_ratio={args.on_off_ratio}",
        fontsize=10, y=1.01,
    )

    # row 0: accumulated train
    ax = axes[0]
    ax.fill_between(t_ms, train_accumulated, alpha=0.35, color="grey", linewidth=0)
    ax.plot(t_ms, train_accumulated, color="grey", linewidth=0.8)
    ax.set_ylabel("pRF amp\n(spikes/s)", fontsize=8)
    ax.set_title("Accumulated boxcar train  (steps 1–5 output)", fontsize=9)
    ax.set_ylim(bottom=0)
    _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    # rows 1…N: per-tone ON + OFF (contrasting colors from colormap ends)
    color_on  = auditory_cortex_cmap(0.85)   # bright orange-red
    color_off = auditory_cortex_cmap(0.15)  # bright teal
    for s in range(n_tones):
        ax = axes[s + 1]
        ax.fill_between(t_ms, per_tone_trains[s], alpha=0.15, color="grey", linewidth=0)
        ax.plot(t_ms, per_tone_on[s], color=color_on, linewidth=1.2,
                label=f"ON  tone {s+1}")
        ax.plot(t_ms, per_tone_off[s], color=color_off, linewidth=1.2,
                linestyle="--", label=f"OFF tone {s+1}")
        ax.set_ylabel(f"Tone {s + 1}\n(spikes/s)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
        _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    # last row: superposed ON + OFF + combined
    combined_response = args.on_off_ratio * on_response + off_response
    ax = axes[-1]
    ax.plot(t_ms, on_response, color=color_on, linewidth=1.0, label="ON")
    ax.plot(t_ms, off_response, color=color_off, linewidth=1.0, linestyle="--", label="OFF")
    ax.plot(t_ms, combined_response, color="#7E2F8E", linewidth=1.4,
            label=f"Combined (on_off_ratio={args.on_off_ratio})")
    ax.set_ylabel("Superposed\n(spikes/s)", fontsize=8)
    ax.set_title("Superposed AdapTrans ON + OFF response  (pipeline output)", fontsize=9)
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    plt.tight_layout()

    # ── 9. Save outputs via ResultSaver ───────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem  = _build_output_stem(seq_id, cf_index, cf_hz,
                               args.alpha, args.pref_dur, args.sigma_dur,
                               args.w, args.on_off_ratio)
    saver = ResultSaver(out_dir)

    # figure
    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved : %s", fig_path)

    # data + metadata
    saver.save_npz(
        data={
            # arrays
            "t_ms":               t_ms,
            "train_accumulated":  train_accumulated,
            "per_tone_trains":    np.stack(per_tone_trains),   # (N_tones, T)
            "per_tone_on":        np.stack(per_tone_on),       # (N_tones, T)
            "per_tone_off":       np.stack(per_tone_off),      # (N_tones, T)
            "on_response":        on_response,
            "off_response":       off_response,
            "combined_response":  combined_response,
            "onsets_ms":          result["onsets_ms"],
            "offsets_ms":         result["offsets_ms"],
            "prf_responses":      np.array(prf_responses),
            "mean_rates_on":      np.array(mean_rates_on),
            # metadata (stored as 0-d object arrays)
            "seq_id":             np.array(seq_id),
            "exp_name":           np.array(args.exp_name),
            "npz_source":         np.array(str(npz_path)),
            "cf_index":           np.array(cf_index),
            "cf_hz":              np.array(cf_hz),
            "alpha":              np.array(args.alpha),
            "pref_dur_ms":        np.array(args.pref_dur),
            "sigma_dur_ms":       np.array(args.sigma_dur),
            "w":                  np.array(args.w),
            "K":                  np.array(-1 if args.K is None else args.K),
            "on_off_ratio":          np.array(args.on_off_ratio),
            "tone_dur_ms":        np.array(tone_dur_ms),
            "isi_ms":             np.array(isi_ms),
            "total_dur_ms":       np.array(total_dur_ms),
            "dt_ms":              np.array(1.0),
        },
        filename=f"{stem}.npz",
    )
    logger.info("Data  saved  : %s", out_dir / f"{stem}.npz")
    logger.info("Done.")


if __name__ == "__main__":
    main()


# ── Usage examples ────────────────────────────────────────────────────────────
# Default (first sequence, CF index 10, all default parameters):
#   python visualize_adaptrans_pertone.py
#
# Custom sequence and ON/OFF weights:
#   python visualize_adaptrans_pertone.py \
#       --exp_name dipc_test_250225_01 \
#       --seq_index 5 --cf 10 \
#       --alpha 2.0 --pref_dur 200 --sigma_dur 20 --w 0.8 \
#       --on_off_ratio 2.0
#
# ON-only combined signal (suppress OFF contribution):
#   python visualize_adaptrans_pertone.py --on_off_ratio 1000.0
#
# Offset-dominated:
#   python visualize_adaptrans_pertone.py --on_off_ratio 0.3

