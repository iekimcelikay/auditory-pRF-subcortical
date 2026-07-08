"""
plot_noiseclouds_bold_timecourses.py
=====================================
Plot predicted BOLD timecourses for the noise-cloud Gaussian-pRF notemporal
pipeline output -- one figure per run, each a grid of 30 CF panels, with a
condition timeline strip on top showing which noise-cloud band / duration was
playing at each point in time.

Usage
-----
    python visualization/scratch/plot_noiseclouds_bold_timecourses.py
    python visualization/scratch/plot_noiseclouds_bold_timecourses.py \\
        --bold_dir models_output/noiseclouds_gaussianprf_20260602_1915/notemporal_bold \\
        --cochlea_dir models_output/noiseclouds_gaussianprf_20260602_1915 \\
        --n_cols 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from auditory_prf.prf_pipeline.run_assembly import generate_run_design
from auditory_prf.prf_pipeline.full_pipeline_noiseclouds_notemporal import (
    BAND_CENTERS_HZ, BW_OCT, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE,
    TONE_ON_MS, ISI_MS, NULL_FRACTION, TRIAL_DURATION_S, OPENING_BLANK_S,
    ITI_RANGE_S, BASE_SEED, NC_SILENCE_SEQ_ID,
    _make_noisecloud_seq_id_fn,
)

_COND_RE = re.compile(r'_fc(\d+)hz_bw[\d.]+oct_dur(\d+)ms')


# ── data loading ──────────────────────────────────────────────────────────────

def load_bold_matrix(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None, int, float]:
    """Load all run timecourses for one CF, plus the noisy variant if present.

    Returns
    -------
    bold_matrix       : np.ndarray, shape (n_runs, n_trs)
    bold_matrix_noisy : np.ndarray, shape (n_runs, n_trs), or None if no
                         ``run_XX_noisy`` keys are present in the npz.
    cf_index          : int
    tr_s              : float
    """
    data = np.load(npz_path, allow_pickle=True)
    run_keys = sorted(
        (k for k in data.files if k.startswith("run_") and not k.endswith("_noisy")),
        key=lambda k: int(k.split("_")[1]),
    )
    bold_matrix = np.stack([data[k] for k in run_keys], axis=0)

    noisy_keys = [f"{k}_noisy" for k in run_keys]
    if all(k in data.files for k in noisy_keys):
        bold_matrix_noisy = np.stack([data[k] for k in noisy_keys], axis=0)
    else:
        bold_matrix_noisy = None

    return bold_matrix, bold_matrix_noisy, int(data["cf"]), float(data["tr_s"])


def load_cf_hz_values(cochlea_dir: Path) -> np.ndarray:
    """Load cf_list (CF in Hz per cf_index) from any cochlear-sim npz.

    Returns
    -------
    cf_hz_values : np.ndarray, shape (n_cfs,)
    """
    npz_files = sorted(cochlea_dir.glob("wav*/**/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No cochlear-sim npz files found under {cochlea_dir}/wav*/")
    data = np.load(npz_files[0], allow_pickle=True)
    return np.asarray(data["cf_list"])


# ── run-design reconstruction ─────────────────────────────────────────────────

def reconstruct_run_designs(n_runs: int, base_seed: int = BASE_SEED) -> list[list[tuple]]:
    """Replay generate_run_design() with this pipeline's default parameters.

    Returns
    -------
    run_designs : list (length n_runs) of list of (seq_id, onset_s)
    """
    n_gaussians = len(BAND_CENTERS_HZ)
    stimuli = [(ton, isi, g_idx)
               for ton, isi in zip(TONE_ON_MS, ISI_MS)
               for g_idx in range(n_gaussians)]
    n_null = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
    base_trials = stimuli + [(0, 0, None)] * n_null

    seq_id_fn = _make_noisecloud_seq_id_fn(
        BAND_CENTERS_HZ, BW_OCT, TOTAL_SEQ_DUR_S, STIMULUS_SAMPLE_RATE
    )

    return [
        generate_run_design(
            base_trials, seq_id_fn,
            trial_duration_s=TRIAL_DURATION_S,
            opening_blank_s=OPENING_BLANK_S,
            iti_range_s=ITI_RANGE_S,
            seed=base_seed + i,
        )
        for i in range(n_runs)
    ]


def parse_condition(seq_id: str) -> tuple[str, int | None]:
    """Return (band_label, duration_ms) for a noise-cloud seq_id, or ("null", None)."""
    if seq_id == NC_SILENCE_SEQ_ID:
        return "null", None
    m = _COND_RE.search(seq_id)
    freq_hz, dur_ms = int(m.group(1)), int(m.group(2))
    return f"{freq_hz} Hz", dur_ms


def band_color_map() -> dict[str, str]:
    """Map band labels (and 'null') to fixed colors for shading/legend."""
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    colors = {f"{hz} Hz": palette[i % len(palette)] for i, hz in enumerate(BAND_CENTERS_HZ)}
    colors["null"] = "lightgrey"
    return colors


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_run_figure(
        cf_data: dict[int, np.ndarray],
        cf_data_noisy: dict[int, np.ndarray] | None,
        cf_hz_values: np.ndarray,
        tr_s: float,
        run_idx: int,
        run_design: list[tuple],
        n_cols: int,
        save_path: Path,
) -> None:
    """One figure for one run: condition timeline strip + grid of CF BOLD panels (with optional noisy overlay)."""
    cf_indices = sorted(cf_data.keys())
    n_cfs  = len(cf_indices)
    n_rows = int(np.ceil(n_cfs / n_cols))
    n_trs  = next(iter(cf_data.values())).shape[1]
    total_run_dur_s = n_trs * tr_s

    colors = band_color_map()
    spans  = [(onset_s, *parse_condition(seq_id)) for seq_id, onset_s in run_design]

    fig = plt.figure(figsize=(4.0 * n_cols, 2.6 * n_rows + 1.2))
    gs  = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.6] + [1] * n_rows, hspace=0.6)

    # ── condition timeline strip ──
    ax_timeline = fig.add_subplot(gs[0, :])
    for onset_s, band_label, dur_ms in spans:
        ax_timeline.axvspan(onset_s, onset_s + TRIAL_DURATION_S,
                             color=colors[band_label], alpha=0.6, lw=0)
        if dur_ms is not None:
            ax_timeline.text(onset_s + TRIAL_DURATION_S / 2, 0.5, f"{dur_ms}",
                              ha="center", va="center", fontsize=6, rotation=90)
    ax_timeline.set_xlim(0, total_run_dur_s)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_ylabel("Condition\n(Hz / dur ms)", fontsize=8)
    ax_timeline.set_title(
        f"Run {run_idx + 1:02d} — condition timeline (band color = freq, number = tone duration in ms)",
        fontsize=10,
    )
    legend_handles = [Patch(facecolor=c, alpha=0.6, label=lbl) for lbl, c in colors.items()]
    if cf_data_noisy:
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color="lightgray", lw=1.5, label="BOLD + noise"))
    ax_timeline.legend(handles=legend_handles, loc="upper right", fontsize=7,
                        ncol=len(legend_handles), framealpha=0.8)

    # ── per-CF BOLD panels ──
    time_s = np.arange(n_trs) * tr_s
    for idx, cf_index in enumerate(cf_indices):
        ax = fig.add_subplot(gs[1 + idx // n_cols, idx % n_cols], sharex=ax_timeline)
        for onset_s, band_label, _ in spans:
            if band_label == "null":
                continue
            ax.axvspan(onset_s, onset_s + TRIAL_DURATION_S,
                       color=colors[band_label], alpha=0.25, lw=0)

        if cf_data_noisy is not None and cf_index in cf_data_noisy:
            ax.plot(time_s, cf_data_noisy[cf_index][run_idx],
                    color="lightgray", lw=0.8, zorder=1, label="BOLD + noise")
        ax.plot(time_s, cf_data[cf_index][run_idx], color="black", lw=1.0, zorder=2, label="BOLD (clean)")
        ax.set_title(f"CF {cf_index:02d}  ({cf_hz_values[cf_index]:.0f} Hz)", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        if idx % n_cols == 0:
            ax.set_ylabel("BOLD (a.u.)", fontsize=9)
        if idx >= n_cfs - n_cols:
            ax.set_xlabel("Time (s)", fontsize=9)

    fig.suptitle(
        "Noise-cloud Gaussian-pRF — predicted BOLD per CF",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-run, per-CF BOLD timecourses with condition timelines."
    )
    parser.add_argument(
        "--bold_dir", default="models_output/noiseclouds_gaussianprf_20260602_1915/notemporal_bold",
        help="Directory containing *_noiseclouds_notemporal_cf*_bold.npz files.",
    )
    parser.add_argument(
        "--cochlea_dir", default=None,
        help="Directory containing wav*/ cochlear-sim results (for CF Hz labels). "
             "Defaults to the parent of --bold_dir.",
    )
    parser.add_argument("--n_cols", type=int, default=5, help="Subplot grid columns (default: 5).")
    parser.add_argument("--save_dir", default=None, help="Where to save figures (default: bold_dir).")
    args = parser.parse_args()

    bold_dir    = Path(args.bold_dir)
    cochlea_dir = Path(args.cochlea_dir) if args.cochlea_dir else bold_dir.parent
    save_dir    = Path(args.save_dir) if args.save_dir else bold_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(bold_dir.glob("*_noiseclouds_notemporal_cf*_bold.npz"))
    if not npz_files:
        print(f"No matching npz files in {bold_dir}")
        return
    print(f"Found {len(npz_files)} CF file(s).")

    cf_data = {}
    cf_data_noisy = {}
    tr_s = 1.6
    for npz_path in npz_files:
        bold_matrix, bold_matrix_noisy, cf_index, tr_s = load_bold_matrix(npz_path)
        cf_data[cf_index] = bold_matrix
        if bold_matrix_noisy is not None:
            cf_data_noisy[cf_index] = bold_matrix_noisy
        print(f"  CF {cf_index:03d} | {bold_matrix.shape[0]} runs x {bold_matrix.shape[1]} TRs")

    cf_hz_values = load_cf_hz_values(cochlea_dir)
    n_runs = next(iter(cf_data.values())).shape[0]
    run_designs = reconstruct_run_designs(n_runs)

    for run_idx in range(n_runs):
        plot_run_figure(
            cf_data, cf_data_noisy or None, cf_hz_values, tr_s, run_idx, run_designs[run_idx],
            n_cols    = args.n_cols,
            save_path = save_dir / f"noiseclouds_bold_timecourses_run{run_idx + 1:02d}.png",
        )


if __name__ == "__main__":
    main()
