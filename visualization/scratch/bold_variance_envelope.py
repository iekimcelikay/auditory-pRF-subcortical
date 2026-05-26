"""
bold_variance_envelope.py
=========================
Plot BOLD mean ± std and 5–95 % band across N run designs for each CF.

Isolates the effect of trial ordering / ITI jitter on the predicted response
by holding the neural model (per-sequence boxcar trains) fixed and varying
only the run structure.

Usage
-----
    python visualization/scratch/bold_variance_envelope.py
    python visualization/scratch/bold_variance_envelope.py \
        --output_dir models_output/prf_notemporal \
        --save_dir figures/bold_variability
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── data loading ──────────────────────────────────────────────────────────────

def load_bold_matrix(npz_path: Path) -> tuple:
    """Load bold_combined across all runs from one CF npz file.

    Parameters
    ----------
    npz_path : Path

    Returns
    -------
    bold_matrix : np.ndarray, shape (n_runs, n_trs)
    tr_s : float
    cf_index : int
    """
    d = np.load(npz_path, allow_pickle=True)
    run_keys = sorted(k for k in d.files if k.startswith("run_"))
    bold_matrix = np.stack([d[k] for k in run_keys], axis=0)
    return bold_matrix, float(d["tr_s"]), int(d["cf"])


# ── statistics ────────────────────────────────────────────────────────────────

def compute_envelope(bold_matrix: np.ndarray) -> dict:
    """Compute variability envelope across runs.

    Parameters
    ----------
    bold_matrix : np.ndarray, shape (n_runs, n_trs)

    Returns
    -------
    dict with keys: mean, std, p5, p95, cv (coeff. of variation)
    """
    mean = bold_matrix.mean(axis=0)
    std  = bold_matrix.std(axis=0)
    return {
        "mean": mean,
        "std":  std,
        "p5":   np.percentile(bold_matrix, 5,  axis=0),
        "p95":  np.percentile(bold_matrix, 95, axis=0),
        "cv":   np.where(mean != 0, std / np.abs(mean), 0.0),
    }


def print_summary(bold_matrix: np.ndarray, tr_s: float, cf_index: int) -> None:
    n_runs, n_trs = bold_matrix.shape
    std_per_tr = bold_matrix.std(axis=0)
    peak_tr    = int(std_per_tr.argmax())
    print(
        f"  CF {cf_index:03d} | {n_runs} runs × {n_trs} TRs | "
        f"peak σ = {std_per_tr.max():.3f} at t = {peak_tr * tr_s:.1f} s | "
        f"mean σ = {std_per_tr.mean():.3f}"
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_envelope(ax: plt.Axes, t_s: np.ndarray, env: dict, cf_index: int,
                  n_runs: int) -> None:
    color = "steelblue"
    ax.fill_between(t_s, env["p5"], env["p95"],
                    alpha=0.20, color=color, label="5–95 %")
    ax.fill_between(t_s, env["mean"] - env["std"], env["mean"] + env["std"],
                    alpha=0.40, color=color, label="mean ± σ")
    ax.plot(t_s, env["mean"], color=color, lw=1.5, label="mean")
    ax.set_ylabel("BOLD (a.u.)")
    ax.set_title(f"CF index {cf_index}  (n = {n_runs} run designs)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cv(ax: plt.Axes, t_s: np.ndarray, env: dict) -> None:
    """Coefficient of variation (std / |mean|) highlights where design matters most."""
    ax.plot(t_s, env["cv"], color="tomato", lw=1.2)
    ax.axhline(env["cv"].mean(), color="tomato", lw=0.8, ls="--",
               label=f"mean CV = {env['cv'].mean():.2f}")
    ax.set_ylabel("CV (σ / |mean|)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BOLD variability envelope across run designs."
    )
    parser.add_argument("--output_dir", default="models_output/prf_notemporal",
                        help="Directory containing *_notemporal_cf*_bold.npz files.")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save figure (default: same as output_dir).")
    parser.add_argument("--show_cv", action="store_true",
                        help="Add coefficient-of-variation panel per CF.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir   = Path(args.save_dir) if args.save_dir else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(output_dir.glob("*_notemporal_cf*_bold.npz"))
    if not npz_files:
        print(f"No matching npz files found in {output_dir}")
        return

    n_cfs    = len(npz_files)
    n_panels = 2 if args.show_cv else 1
    fig, axes = plt.subplots(
        n_cfs * n_panels, 1,
        figsize=(14, 3 * n_cfs * n_panels),
        sharex=True,
        squeeze=False,
    )

    print(f"Found {n_cfs} CF file(s) in {output_dir}")
    for i, npz_path in enumerate(npz_files):
        bold_matrix, tr_s, cf_index = load_bold_matrix(npz_path)
        n_runs, n_trs = bold_matrix.shape
        t_s = np.arange(n_trs) * tr_s

        print_summary(bold_matrix, tr_s, cf_index)
        env = compute_envelope(bold_matrix)

        plot_envelope(axes[i * n_panels, 0], t_s, env, cf_index, n_runs)
        if args.show_cv:
            plot_cv(axes[i * n_panels + 1, 0], t_s, env)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(
        f"BOLD variability across {n_runs} run designs — run structure only\n"
        "inner band: mean ± σ    outer band: 5–95 %",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = save_dir / "bold_variance_envelope.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
