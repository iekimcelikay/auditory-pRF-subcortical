"""
visualize_adaptrans_pertone_impulse.py
======================================
Impulse-train variant of visualize_adaptrans_pertone.py.

.. warning::
    This implementation is kept for conceptual reference but does NOT produce
    the intended temporal dynamics of the AdapTrans ON filter.

    The AdapTrans h_ON kernel is a finite-difference / adaptation filter: it
    compares the current input to an exponentially-weighted average of the
    recent past.  It needs a *sustained* input (boxcar) to express its
    characteristic onset-then-decay response.  Given a delta-spike input the
    kernel fires for exactly **one sample** at onset and is zero everywhere
    else — there is no temporal evolution across the tone duration.

    The algebraic identity  output[n] = sum_s prf[s] * h_ON[n - n_s^onset]
    holds only when h_ON is used as a *response-shape* kernel (analogous to
    an HRF).  That is not what AdapTrans is.

    Use visualize_adaptrans_pertone.py (boxcar input) for correct results.

Each tone contributes a single delta spike at its onset, scaled by its pRF
response scalar, matching the mathematical definition:

    x[n]      = sum_s  prf_response[s] * delta[n - n_s^onset]
    output[n] = (x * h_ON)[n]
              = sum_s  prf_response[s] * h_ON[n - n_s^onset]

Layout (N_tones + 2 rows):
  Row 0          : accumulated impulse train (all tones summed, for reference)
  Rows 1–N_tones : per-tone isolated impulse (stem) + its ON response (line)
  Row N_tones+1  : superposed ON response (sum of all per-tone ON responses)

Outputs saved to <out_dir>/:
  <stem>.png  — figure
  <stem>.npz  — all arrays + metadata

Usage
-----
    python visualize_adaptrans_pertone_impulse.py
    python visualize_adaptrans_pertone_impulse.py --exp_name dipc_test_250225_01 \\
        --seq_index 5 --cf 10 --alpha 2.0 --pref_dur 200 --sigma_dur 20 --w 0.8
    python visualize_adaptrans_pertone_impulse.py --out_dir figures/adaptrans_impulse_check
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
DEFAULT_OUT_DIR  = Path(f"./figures/{EXP_NAME}/adaptrans_pertone_impulse")

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise per-tone AdapTrans ON filter (impulse-train variant)."
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
        f"adaptrans_pertone_impulse_{seq_id}"
        f"_cfidx{cf_index}_cfhz{cf_hz:.0f}"
        f"_alpha{alpha:.1f}_tau{pref_dur:.0f}"
        f"_sigma{sigma_dur:.0f}_w{w:.2f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    base_dir = Path(f"./models_output/{args.exp_name}")
    out_dir  = (args.out_dir or Path(f"./figures/{args.exp_name}/adaptrans_pertone_impulse")).expanduser().resolve()

    LoggingConfigurator(
        output_dir=out_dir,
        log_filename="visualize_adaptrans_pertone_impulse.log",
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
    population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, args.cf)
    dt_s         = time_axis[1] - time_axis[0]
    total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
    n_1ms        = math.ceil(total_dur_ms)
    t_ms         = np.arange(n_1ms, dtype=float)
    logger.info("CF index: %d | CF: %.1f Hz | seq_id: %s", cf_index, cf_hz, seq_id)
    logger.info("total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

    # ── 3. Power-law sharpening ───────────────────────────────────────────────
    sharpened_pop = apply_powerlaw_population(population_psth, args.alpha)
    sharpened     = sharpened_pop[cf_index, :]
    logger.debug("Sharpened timecourse shape: %s", sharpened.shape)

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

    # ── 6-7. Per-tone impulse train → AdapTrans ON → superpose ───────────────
    per_tone_trains   = []
    per_tone_on       = []
    on_response       = np.zeros(n_1ms)
    train_accumulated = np.zeros(n_1ms)

    for s, (prf_s, on_ms) in enumerate(
            zip(prf_responses, result["onsets_ms"])):

        single_train = build_prf_impulse_train(
            [prf_s], np.array([on_ms]),
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
        on_response += per_tone_on[-1]
        logger.debug("  Tone %d/%d | ON max: %.4e", s + 1, n_tones, per_tone_on[-1].max())

    logger.info("ON response shape: %s | max: %.4e", on_response.shape, on_response.max())

    # ── 8. Figure ─────────────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_tones))
    n_rows = 1 + n_tones + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.4 * n_rows), sharex=True)
    fig.suptitle(
        f"{seq_id}  [impulse-train]\n"
        f"CF={cf_hz:.1f} Hz (idx {cf_index}) | "
        f"α={args.alpha} | τ₀={args.pref_dur:.0f} ms | σ={args.sigma_dur:.0f} ms | "
        f"w={args.w} | K={'auto' if args.K is None else args.K}",
        fontsize=10, y=1.01,
    )

    # row 0: accumulated impulse train (shown as stems)
    ax = axes[0]
    onset_indices = [round(on / 1.0) for on in result["onsets_ms"]]
    onset_amps    = [train_accumulated[i] for i in onset_indices]
    ax.vlines(result["onsets_ms"], 0, onset_amps, color="grey", linewidth=1.2)
    ax.plot(result["onsets_ms"], onset_amps, "o", color="grey", markersize=4)
    ax.set_ylabel("pRF amp\n(spikes/s)", fontsize=8)
    ax.set_title("Accumulated impulse train  (steps 1–5 output)", fontsize=9)
    ax.set_ylim(bottom=0)
    _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    # rows 1…N: per-tone impulse (stem) + ON response (line)
    for s in range(n_tones):
        ax = axes[s + 1]
        c  = colors[s]
        on_i = round(result["onsets_ms"][s] / 1.0)
        amp  = per_tone_trains[s][on_i]
        ax.vlines(result["onsets_ms"][s], 0, amp, color=c, linewidth=1.5, alpha=0.6)
        ax.plot(result["onsets_ms"][s], amp, "o", color=c, markersize=5, alpha=0.6)
        ax.plot(t_ms, per_tone_on[s], color=c, linewidth=1.2, label=f"ON  tone {s+1}")
        ax.set_ylabel(f"Tone {s + 1}\nON (spikes/s)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
        _add_tone_markers(ax, result["onsets_ms"], result["offsets_ms"], n_1ms)

    # last row: superposed ON response
    ax = axes[-1]
    ax.plot(t_ms, on_response, color="black", linewidth=1.2)
    ax.set_ylabel("Superposed ON\n(spikes/s)", fontsize=8)
    ax.set_title("Superposed AdapTrans ON response  (pipeline output)", fontsize=9)
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
            "t_ms":               t_ms,
            "train_accumulated":  train_accumulated,
            "per_tone_trains":    np.stack(per_tone_trains),   # (N_tones, T)
            "per_tone_on":        np.stack(per_tone_on),       # (N_tones, T)
            "on_response":        on_response,
            "onsets_ms":          result["onsets_ms"],
            "offsets_ms":         result["offsets_ms"],
            "prf_responses":      np.array(prf_responses),
            "mean_rates_on":      np.array(mean_rates_on),
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
#   python visualize_adaptrans_pertone_impulse.py
#
# Custom sequence and parameters:
#   python visualize_adaptrans_pertone_impulse.py \
#       --exp_name dipc_test_250225_01 \
#       --seq_index 5 --cf 10 \
#       --alpha 2.0 --pref_dur 200 --sigma_dur 20 --w 0.8
#
# Custom output directory:
#   python visualize_adaptrans_pertone_impulse.py --out_dir figures/adaptrans_impulse_check
