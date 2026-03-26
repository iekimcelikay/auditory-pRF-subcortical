"""
visualize_adaptrans_shapekern.py
=================================
Visualise the response-shape kernel convolution approach to AdapTrans ON.

The AdapTrans response to a unit boxcar of duration D is used as a
response-shape kernel h_ON_shape.  A scaled impulse train is then convolved
with this kernel in a single pass, equivalent to the per-tone boxcar loop:

    h_ON_shape  = AdapTrans_ON( unit_boxcar[0:D] )    # computed once
    x[n]        = sum_s  prf_response[s] * delta[n - n_s^onset]
    output[n]   = (x * h_ON_shape)[n]
                = sum_s  prf_response[s] * h_ON_shape[n - n_s^onset]

This is mathematically equivalent to the per-tone boxcar loop when all tones
share the same duration (true for pure-tone DIPC sequences).

Layout (3 rows):
  Row 0 : h_ON_shape kernel (unit-boxcar AdapTrans ON response)
  Row 1 : scaled impulse train x[n]  (pRF amplitude at each tone onset)
  Row 2 : output[n] = x * h_ON_shape  (final ON timecourse)

Outputs saved to <out_dir>/:
  <stem>.png  — figure
  <stem>.npz  — arrays + metadata

Usage
-----
    python visualize_adaptrans_shapekern.py
    python visualize_adaptrans_shapekern.py --exp_name dipc_test_250225_01 \\
        --seq_index 5 --cf 10 --alpha 2.0 --pref_dur 200 --sigma_dur 20 --w 0.8
    python visualize_adaptrans_shapekern.py --out_dir figures/adaptrans_shapekern
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from auditory_prf.utils.cochlea_loader_functions import resolve_results_dir
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import (
    build_prf_impulse_train,
    apply_adaptrans,
)

# ── defaults ─────────────────────────────────────────────────────────────────
EXP_NAME         = "dipc_test_250225_01"
DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")
DEFAULT_OUT_DIR  = Path(f"./figures/{EXP_NAME}/adaptrans_shapekern")

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise AdapTrans ON via response-shape kernel convolution."
    )
    p.add_argument("--exp_name",    type=str,   default=EXP_NAME)
    p.add_argument("--results_dir", type=Path,  default=None,
                   help="Path to the .npz results folder (or parent).")
    p.add_argument("--out_dir",     type=Path,  default=None,
                   help=f"Output directory. Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--seq_index",   type=int,   default=0,
                   help="0-based index into the sorted list of .npz files (default 0).")
    p.add_argument("--cf",          type=int,   default=10,
                   help="CF index to extract from the population PSTH (default 10).")
    p.add_argument("--alpha",       type=float, default=2.0,
                   help="Power-law sharpening exponent (default 2.0).")
    p.add_argument("--pref_dur",    type=float, default=200.0,
                   help="Preferred duration for the Gaussian kernel in ms (default 200).")
    p.add_argument("--sigma_dur",   type=float, default=20.0,
                   help="Sigma of the duration Gaussian in ms (default 20).")
    p.add_argument("--w",           type=float, default=0.8,
                   help="AdapTrans adaptation weight (default 0.8).")
    p.add_argument("--K",           type=int,   default=None,
                   help="AdapTrans kernel length in samples (auto if omitted).")
    p.add_argument("--dpi",         type=int,   default=150)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _add_tone_markers(ax, onsets_ms, offsets_ms, n_1ms):
    """Shade each tone-on period in light blue."""
    for on, off in zip(onsets_ms, offsets_ms):
        ax.axvspan(on, min(off, n_1ms), alpha=0.10, color="steelblue",
                   linewidth=0, zorder=0)


def _build_output_stem(seq_id, cf_index, cf_hz, alpha, pref_dur, sigma_dur, w):
    return (
        f"adaptrans_shapekern_{seq_id}"
        f"_cfidx{cf_index}_cfhz{cf_hz:.0f}"
        f"_alpha{alpha:.1f}_tau{pref_dur:.0f}"
        f"_sigma{sigma_dur:.0f}_w{w:.2f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    base_dir = Path(f"./models_output/{args.exp_name}")
    out_dir  = (args.out_dir or Path(f"./figures/{args.exp_name}/adaptrans_shapekern")).expanduser().resolve()

    LoggingConfigurator(
        output_dir=out_dir,
        log_filename="visualize_adaptrans_shapekern.log",
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

    # ── 2. Load population PSTH ───────────────────────────────────────────────
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(
        npz_path, args.cf
    )
    dt_s         = time_axis[1] - time_axis[0]
    total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
    n_1ms        = math.ceil(total_dur_ms)
    t_ms         = np.arange(n_1ms, dtype=float)
    logger.info("CF index: %d | CF: %.1f Hz | seq_id: %s", cf_index, cf_hz, seq_id)
    logger.info("total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

    # ── 3. Power-law sharpening ───────────────────────────────────────────────
    sharpened_pop = apply_powerlaw_population(population_psth, args.alpha)
    sharpened     = sharpened_pop[cf_index, :]

    # ── 4. Chunk into tone-on windows ─────────────────────────────────────────
    result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id)
    n_tones       = len(result["chunks"])
    mean_rates_on = [np.mean(c) for c in result["chunks"]]
    logger.info("tone_dur: %.1f ms | ISI: %.1f ms | n_tones: %d",
                tone_dur_ms, isi_ms, n_tones)

    # ── 5. Duration Gaussian (scalar) ─────────────────────────────────────────
    prf_responses = [
        apply_duration_gaussian_scalar(m, tone_dur_ms, args.pref_dur, args.sigma_dur)
        for m in mean_rates_on
    ]
    logger.info("pRF responses: %s", [f"{p:.3e}" for p in prf_responses])

    # ── 6. Compute response-shape kernel ─────────────────────────────────────
    # Pass a unit boxcar of length tone_dur_ms through AdapTrans ON once.
    # The output is the temporal profile that each tone's response will follow.
    unit_boxcar = np.zeros(n_1ms)
    i_off_kern  = round(tone_dur_ms)
    unit_boxcar[0:i_off_kern] = 1.0

    on_off_unit = apply_adaptrans(
        unit_boxcar[np.newaxis, :],
        CFs_Hz=np.array([cf_hz]),
        dt_ms=1.0,
        w=args.w,
        K=args.K,
        pad_value=0.0,
    )
    h_ON_shape = on_off_unit[0, 0, :]   # shape (n_1ms,)
    logger.info("h_ON_shape: max=%.4e | non-zero samples=%d",
                h_ON_shape.max(), np.count_nonzero(h_ON_shape))

    # ── 7. Build impulse train and convolve ───────────────────────────────────
    impulse_train = build_prf_impulse_train(
        prf_responses, result["onsets_ms"],
        total_dur_ms, dt_ms=1.0,
    )

    on_response = np.convolve(impulse_train, h_ON_shape, mode='full')[:n_1ms]
    logger.info("ON response: max=%.4e", on_response.max())

    # ── 8. Figure ─────────────────────────────────────────────────────────────
    # t axis for the kernel (relative to onset, trimmed to non-trivial length)
    kern_len   = np.flatnonzero(h_ON_shape)[-1] + 1 if h_ON_shape.any() else n_1ms
    t_kern_ms  = np.arange(kern_len, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f"{seq_id}  [response-shape kernel]\n"
        f"CF={cf_hz:.1f} Hz (idx {cf_index}) | "
        f"α={args.alpha} | τ₀={args.pref_dur:.0f} ms | σ={args.sigma_dur:.0f} ms | "
        f"w={args.w} | K={'auto' if args.K is None else args.K}",
        fontsize=10, y=1.01,
    )

    # row 0: h_ON_shape kernel
    ax = axes[0]
    ax.fill_between(t_kern_ms, h_ON_shape[:kern_len], alpha=0.30,
                    color="darkorange", linewidth=0)
    ax.plot(t_kern_ms, h_ON_shape[:kern_len], color="darkorange", linewidth=1.2)
    ax.axvspan(0, tone_dur_ms, alpha=0.12, color="steelblue", linewidth=0,
               label=f"unit boxcar ({tone_dur_ms:.0f} ms)")
    ax.set_ylabel("h_ON_shape\n(a.u.)", fontsize=8)
    ax.set_xlabel("Time from tone onset (ms)", fontsize=8)
    ax.set_title("Response-shape kernel  h_ON_shape = AdapTrans_ON( unit boxcar )", fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    # row 1: scaled impulse train
    ax = axes[1]
    ax.sharex(axes[2])
    onset_amps = [impulse_train[round(on)] for on in result["onsets_ms"]]
    ax.vlines(result["onsets_ms"], 0, onset_amps,
              color="steelblue", linewidth=1.5, label="pRF-weighted onset")
    ax.plot(result["onsets_ms"], onset_amps, "o", color="steelblue", markersize=5)
    ax.set_ylabel("x[n]  pRF amp\n(spikes/s)", fontsize=8)
    ax.set_title("Scaled impulse train  x[n] = Σ prf[s]·δ[n − n_onset]", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    # row 2: output = x * h_ON_shape
    ax = axes[2]
    ax.plot(t_ms, on_response, color="black", linewidth=1.2)
    ax.set_ylabel("output[n]\n(spikes/s)", fontsize=8)
    ax.set_title("output[n] = (x * h_ON_shape)[n]", fontsize=9)
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(bottom=0)
    _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    plt.tight_layout()

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem  = _build_output_stem(seq_id, cf_index, cf_hz,
                               args.alpha, args.pref_dur, args.sigma_dur, args.w)
    saver = ResultSaver(out_dir)

    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved : %s", fig_path)

    saver.save_npz(
        data={
            "t_ms":           t_ms,
            "h_ON_shape":     h_ON_shape,
            "impulse_train":  impulse_train,
            "on_response":    on_response,
            "onsets_ms":      result["onsets_ms"],
            "offsets_ms":     result["offsets_ms"],
            "prf_responses":  np.array(prf_responses),
            "mean_rates_on":  np.array(mean_rates_on),
            "seq_id":         np.array(seq_id),
            "exp_name":       np.array(args.exp_name),
            "npz_source":     np.array(str(npz_path)),
            "cf_index":       np.array(cf_index),
            "cf_hz":          np.array(cf_hz),
            "alpha":          np.array(args.alpha),
            "pref_dur_ms":    np.array(args.pref_dur),
            "sigma_dur_ms":   np.array(args.sigma_dur),
            "w":              np.array(args.w),
            "K":              np.array(-1 if args.K is None else args.K),
            "tone_dur_ms":    np.array(tone_dur_ms),
            "isi_ms":         np.array(isi_ms),
            "total_dur_ms":   np.array(total_dur_ms),
            "dt_ms":          np.array(1.0),
        },
        filename=f"{stem}.npz",
    )
    logger.info("Data  saved  : %s", out_dir / f"{stem}.npz")
    logger.info("Done.")


if __name__ == "__main__":
    main()


# ── Usage examples ────────────────────────────────────────────────────────────
# Default (first sequence, CF index 10, all default parameters):
#   python visualize_adaptrans_shapekern.py
#
# Custom sequence and parameters:
#   python visualize_adaptrans_shapekern.py \
#       --exp_name dipc_test_250225_01 \
#       --seq_index 5 --cf 10 \
#       --alpha 2.0 --pref_dur 200 --sigma_dur 20 --w 0.8
#
# Custom output directory:
#   python visualize_adaptrans_shapekern.py --out_dir figures/adaptrans_shapekern
