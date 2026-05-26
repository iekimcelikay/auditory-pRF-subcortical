"""
plot_design_efficiency.py
=========================
Compute and plot GLM design efficiency for all ranked run designs.

Efficiency = n_conditions / trace((X'X)^-1)
where X is the HRF-convolved design matrix (one column per condition).

Efficiency is CF-independent — it depends only on trial timing.

Usage
-----
    python visualization/scratch/plot_design_efficiency.py
    python visualization/scratch/plot_design_efficiency.py \
        --output_dir models_output/prf_notemporal_20260518_job5634143 --top_n 20
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parents[2]))

from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.run_assembly import make_seq_id_fn, generate_run_design
from auditory_prf.utils.stimulus_utils import calc_cfs

# ── must match run_pipeline_notemporal.py ─────────────────────────────────────
BASE_SEED            = 42
ALL_DURATIONS        = (30, 50, 75, 110, 150, 200, 350, 450)
ISI_MS               = 100
FREQ_RANGE           = (400, 1600, 3)
TRIAL_DURATION_S     = 20.0
OPENING_BLANK_S      = 10.0
ITI_RANGE_S          = (0, 0)
NULL_FRACTION        = 0.25
N_CONDITIONS_PER_RUN = len(ALL_DURATIONS)
TR_S                 = 1.6
N_TRS                = 450


# ── HRF at TR resolution ─────────────────────────────────────────────────────

def _build_hrf_tr(tr_s: float = TR_S) -> np.ndarray:
    """Double-gamma HRF sampled at TR resolution (subcortical params)."""
    kernel_1ms, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=1e-3, duration=32.0)
    # downsample: take every (tr_s / 1e-3)-th sample
    step = int(round(tr_s / 1e-3))
    return kernel_1ms[::step]


# ── condition index table ─────────────────────────────────────────────────────

def _build_condition_table(desired_freqs):
    """Return sorted (dur_ms, freq_hz) pairs and a lookup dict → col index.

    Frequencies are rounded to int to match the integer values encoded in seq_ids
    (e.g. calc_cfs returns 830.301... but seq_ids contain 'fc830hz').
    """
    pairs = sorted(
        [(dur, int(round(freq))) for dur in ALL_DURATIONS for freq in desired_freqs]
    )
    lookup = {p: i for i, p in enumerate(pairs)}
    return pairs, lookup


def _parse_seq_id(seq_id) -> tuple:
    if seq_id is None or seq_id == "null":
        return None, None
    freq_hz     = int(re.search(r'_fc(\d+)hz',  seq_id).group(1))
    duration_ms = int(re.search(r'_dur(\d+)ms', seq_id).group(1))
    return freq_hz, duration_ms


# ── design matrix builder ─────────────────────────────────────────────────────

def build_design_matrix(run_design: list, condition_lookup: dict,
                        hrf_tr: np.ndarray,
                        n_trs: int = N_TRS,
                        tr_s: float = TR_S,
                        trial_dur_s: float = TRIAL_DURATION_S) -> np.ndarray:
    """Build HRF-convolved design matrix for one run.

    Parameters
    ----------
    run_design : list of (seq_id, onset_s)

    Returns
    -------
    X : np.ndarray, shape (n_trs, n_conditions)
        Mean-centred, HRF-convolved design matrix.
    """
    n_cond = len(condition_lookup)
    trial_dur_trs = max(1, int(round(trial_dur_s / tr_s)))

    # boxcar matrix at TR resolution
    X_box = np.zeros((n_trs, n_cond))
    for seq_id, onset_s in run_design:
        freq_hz, dur_ms = _parse_seq_id(seq_id)
        if freq_hz is None:
            continue
        key = (dur_ms, freq_hz)
        if key not in condition_lookup:
            continue
        col = condition_lookup[key]
        onset_tr = int(round(float(onset_s) / tr_s))
        end_tr   = min(onset_tr + trial_dur_trs, n_trs)
        if onset_tr < n_trs:
            X_box[onset_tr:end_tr, col] = 1.0

    # convolve each column with HRF and truncate
    X_hrf = np.zeros_like(X_box)
    for col in range(n_cond):
        conv = np.convolve(X_box[:, col], hrf_tr)
        X_hrf[:, col] = conv[:n_trs]

    # mean-centre each regressor (remove intercept confound)
    X_hrf -= X_hrf.mean(axis=0)
    return X_hrf


def compute_efficiency(X: np.ndarray) -> float:
    """A-efficiency: n_conditions / trace((X'X)^-1).

    Returns 0.0 if matrix is singular (degenerate design).
    """
    n_cond = X.shape[1]
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
        return float(n_cond / np.trace(XtX_inv))
    except np.linalg.LinAlgError:
        return 0.0


# ── reconstruct run design from seed ─────────────────────────────────────────

def reconstruct_run_design(run_number: int, seq_id_fn,
                            desired_freqs) -> list:
    i   = run_number - 1
    rng = np.random.default_rng(BASE_SEED + i)
    sampled_durations = rng.choice(ALL_DURATIONS, size=N_CONDITIONS_PER_RUN,
                                   replace=False)
    stimuli     = [(int(dur), ISI_MS, freq)
                   for dur in sampled_durations
                   for freq in desired_freqs]
    n_null      = int(np.floor(len(stimuli) * NULL_FRACTION / (1 - NULL_FRACTION)))
    base_trials = stimuli + [(0, 0, None)] * n_null
    design_seed = int(rng.integers(0, 2**32))
    return generate_run_design(
        base_trials, seq_id_fn,
        trial_duration_s = TRIAL_DURATION_S,
        opening_blank_s  = OPENING_BLANK_S,
        iti_range_s      = ITI_RANGE_S,
        seed             = design_seed,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="models_output/prf_notemporal_20260518_job5634143")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--n_sample", type=int, default=500,
                        help="Designs sampled for the distribution (for speed). "
                             "Set to 0 to compute all 5000.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    d = np.load(output_dir / "design_ranking.npz", allow_pickle=True)
    ranked_run_numbers = d["ranked_run_numbers"]
    distances          = d["distances"]
    n_total            = len(ranked_run_numbers)

    print("Building helpers…")
    desired_freqs  = calc_cfs(FREQ_RANGE, species='human')
    seq_id_fn      = make_seq_id_fn(FREQ_RANGE,
                                    tone_on_ms_options=ALL_DURATIONS,
                                    isi_ms_options=(ISI_MS,) * len(ALL_DURATIONS))
    condition_pairs, condition_lookup = _build_condition_table(desired_freqs)
    hrf_tr = _build_hrf_tr(TR_S)
    n_cond = len(condition_pairs)
    print(f"  {n_cond} conditions | HRF length {len(hrf_tr)} TRs")

    # ── which designs to evaluate ──────────────────────────────────────────
    best_runs  = ranked_run_numbers[:args.top_n]
    worst_runs = ranked_run_numbers[-args.top_n:]

    if args.n_sample > 0:
        rng_sample = np.random.default_rng(0)
        sample_idx = rng_sample.choice(n_total, size=min(args.n_sample, n_total),
                                        replace=False)
        sample_runs = ranked_run_numbers[sample_idx]
        sample_dist = distances[sample_idx]
    else:
        sample_runs = ranked_run_numbers
        sample_dist = distances

    all_runs_to_eval = np.unique(
        np.concatenate([best_runs, worst_runs, sample_runs])
    )

    print(f"Computing efficiency for {len(all_runs_to_eval)} designs…")
    efficiency_map: dict = {}
    for i, run_num in enumerate(all_runs_to_eval):
        design = reconstruct_run_design(int(run_num), seq_id_fn, desired_freqs)
        X      = build_design_matrix(design, condition_lookup, hrf_tr)
        efficiency_map[int(run_num)] = compute_efficiency(X)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_runs_to_eval)}")

    eff_best   = np.array([efficiency_map[int(r)] for r in best_runs])
    eff_worst  = np.array([efficiency_map[int(r)] for r in worst_runs])
    eff_sample = np.array([efficiency_map[int(r)] for r in sample_runs
                           if int(r) in efficiency_map])
    dist_sample = np.array([distances[np.where(ranked_run_numbers == r)[0][0]]
                             for r in sample_runs if int(r) in efficiency_map])

    # ── plots ────────────────────────────────────────────────────────────
    out_path = output_dir / "design_efficiency.pdf"
    with PdfPages(out_path) as pdf:

        # Page 1: efficiency distribution + best vs worst comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.hist(eff_sample, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
        for e in eff_best:
            ax.axvline(e, color="royalblue", lw=0.8, alpha=0.6)
        for e in eff_worst:
            ax.axvline(e, color="firebrick", lw=0.8, alpha=0.6)
        ax.axvline(eff_best.mean(),  color="royalblue", lw=2,
                   label=f"Best {args.top_n} mean = {eff_best.mean():.3f}")
        ax.axvline(eff_worst.mean(), color="firebrick",  lw=2,
                   label=f"Worst {args.top_n} mean = {eff_worst.mean():.3f}")
        ax.set_xlabel("Efficiency", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Efficiency distribution\n(sampled designs)", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[1]
        ranks  = np.arange(1, args.top_n + 1)
        ax.bar(ranks - 0.2, eff_best,  width=0.35, color="royalblue",
               label=f"Best {args.top_n} (low distance)")
        ax.bar(ranks + 0.2, eff_worst, width=0.35, color="firebrick",
               label=f"Worst {args.top_n} (high distance)")
        ax.set_xlabel("Rank", fontsize=9)
        ax.set_ylabel("Efficiency", fontsize=9)
        ax.set_title("Efficiency: best vs worst designs", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        ax = axes[2]
        ax.scatter(dist_sample, eff_sample, s=8, alpha=0.4, color="steelblue",
                   label="Sampled")
        ax.scatter(distances[:args.top_n], eff_best, s=30, color="royalblue",
                   zorder=5, label=f"Best {args.top_n}")
        ax.scatter(distances[-args.top_n:], eff_worst, s=30, color="firebrick",
                   marker="^", zorder=5, label=f"Worst {args.top_n}")
        ax.set_xlabel("Distance from mean BOLD", fontsize=9)
        ax.set_ylabel("Efficiency", fontsize=9)
        ax.set_title("Distance vs Efficiency\n(are they correlated?)", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        fig.suptitle("Design matrix efficiency across run designs", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: per-condition regressor correlation matrix for best vs worst #1
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, run_num, label, cmap in zip(
                axes,
                [int(best_runs[0]), int(worst_runs[0])],
                [f"Best #1 (run_{int(best_runs[0]):04d})",
                 f"Worst #1 (run_{int(worst_runs[0]):04d})"],
                ["Blues", "Reds"],
        ):
            design = reconstruct_run_design(run_num, seq_id_fn, desired_freqs)
            X      = build_design_matrix(design, condition_lookup, hrf_tr)
            corr   = np.corrcoef(X.T)
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{label}\neff = {efficiency_map[run_num]:.4f}", fontsize=9)
            ax.set_xlabel("Condition", fontsize=8)
            ax.set_ylabel("Condition", fontsize=8)
            plt.colorbar(im, ax=ax, label="Correlation")

        fig.suptitle("Regressor correlation matrix\n"
                     "(high off-diagonal = low efficiency)", fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved to {out_path}")
    print(f"\nBest {args.top_n}  — efficiency: "
          f"mean={eff_best.mean():.4f}  std={eff_best.std():.4f}")
    print(f"Worst {args.top_n} — efficiency: "
          f"mean={eff_worst.mean():.4f}  std={eff_worst.std():.4f}")
    print(f"Sample    — efficiency: "
          f"mean={eff_sample.mean():.4f}  std={eff_sample.std():.4f}")


if __name__ == "__main__":
    main()
