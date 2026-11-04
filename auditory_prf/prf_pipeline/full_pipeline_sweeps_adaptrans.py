"""full_pipeline_sweeps_adaptrans.py
====================================
pRF pipeline stages for continuous FM-sweep stimuli: power-law sharpening, then AdapTrans.

The cochlear population PSTH (n_cf, n_bins) goes through the pipeline as one continuous
array: no chunking into tone windows, no per-tone mean, no duration filter and no boxcar
or impulse train. Stages (per stimulus):
  1. Load the combined CochleaWavSimulation .npz (all CFs, full 20 s incl. final silence)
  2. Power-law sharpening: x ** alpha, rescaled with one population factor computed over
     the pool of all stimuli, so mean(pool) is preserved (recomputed for every alpha)
  3. AdapTrans ON/OFF on the whole sharpened array; tau_ms (and tau_ms_off) is a single
     free parameter, independent of CF

Every stage output is saved per stimulus, so the stages can be inspected separately.

Run from the project root:
    python auditory_prf/prf_pipeline/full_pipeline_sweeps_adaptrans.py \
        --results-dir models_output/sweep_design_erb_3band/combined

--exclude-n-cycles N [N ...] skips stimuli with those cycle counts; the silent trial
(silence_20s) has n_cycles = 0 in its metadata.
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.prf_pipeline.adaptrans_onoff_filters import (
    build_OFF_kernel,
    build_ON_kernel,
    tau_to_a,
)
from auditory_prf.prf_pipeline.powerlaw_function import (
    apply_powerlaw_population,
    compute_population_scale,
)
from auditory_prf.utils.logging_configurator import LoggingConfigurator

logger = logging.getLogger(__name__)

# ── Defaults (same values as the tone-cloud AdapTrans pipeline) ───────────────
ALPHA = 2.0
ADAPTRANS_W = 0.8
ADAPTRANS_K = None           # auto: 3 x longest tau, in samples
ADAPTRANS_RECTIFY = True
ADAPTRANS_TAU_MS = 100.0     # ON-filter time constant; free parameter, same for all CFs
ADAPTRANS_TAU_MS_OFF = None  # OFF-filter time constant; None = same as tau_ms
PLOT_SECONDS = 10.0


# ── Loading ───────────────────────────────────────────────────────────────────
def load_sweep_result(npz_path: Path) -> dict:
    """Load one combined CochleaWavSimulation .npz, keeping every CF.

    Returns
    -------
    dict with
        soundfileid (str), metadata (dict),
        cf_list (n_cf,) Hz, time_axis (n_bins,) s,
        population_rate_psth (n_cf, n_bins) spikes/s
    """
    with np.load(npz_path, allow_pickle=True) as data:
        result = {
            "soundfileid": str(data["soundfileid"]),
            "metadata": data["metadata"].item(),
            "cf_list": data["cf_list"],
            "time_axis": data["time_axis"],
            "population_rate_psth": data["population_rate_psth"],
        }
    n_cf, n_bins = result["population_rate_psth"].shape
    assert result["cf_list"].shape == (n_cf,), "cf_list does not match PSTH rows"
    assert result["time_axis"].shape == (n_bins,), "time_axis does not match PSTH columns"
    return result


def select_files(npz_files: list, exclude_n_cycles: tuple = ()) -> list:
    """Drop stimuli whose metadata n_cycles is in exclude_n_cycles."""
    kept = []
    for path in npz_files:
        with np.load(path, allow_pickle=True) as data:
            n_cycles = data["metadata"].item().get("n_cycles")
        if n_cycles in exclude_n_cycles:
            logger.info("Excluding %s (n_cycles=%s)", path.name, n_cycles)
            continue
        kept.append(path)
    return kept


def iter_population_psths(npz_files: list) -> Iterator[np.ndarray]:
    """Stream the raw (n_cf, n_bins) population PSTH of every file (for the pool scale)."""
    for path in npz_files:
        yield load_sweep_result(path)["population_rate_psth"]


# ── Analysis ──────────────────────────────────────────────────────────────────
def apply_adaptrans_continuous(
    signal: np.ndarray,
    dt_ms: float,
    w: float = ADAPTRANS_W,
    K: Optional[int] = ADAPTRANS_K,
    rectify: bool = ADAPTRANS_RECTIFY,
    tau_ms: float = ADAPTRANS_TAU_MS,
    tau_ms_off: Optional[float] = ADAPTRANS_TAU_MS_OFF,
) -> tuple:
    """AdapTrans ON/OFF on continuous (n_cf, n_bins) signals, one time constant for all CFs.

    Same kernels, causal first-sample padding and rectification as
    adaptrans_onoff_filters.apply_adaptrans (checked against it in the tests), without
    its boxcar-specific debug indexing, which raises IndexError when a row is positive in
    almost every bin (a continuous PSTH). adaptrans_onoff_filters.py stays untouched.

    Returns
    -------
    on_response, off_response : np.ndarray, each (n_cf, n_bins)
    K : int, the kernel length used
    """
    n_cf, n_bins = signal.shape
    tau_ms_off_used = tau_ms if tau_ms_off is None else tau_ms_off
    if K is None:   # same rule as apply_adaptrans: 3 x the longer tau
        K = int(np.ceil(3 * max(tau_ms, tau_ms_off_used) / dt_ms))
    kernel_on = build_ON_kernel(tau_to_a(tau_ms, dt_ms), w, K)
    kernel_off = build_OFF_kernel(tau_to_a(tau_ms_off_used, dt_ms), w, K)

    on_response = np.zeros((n_cf, n_bins))
    off_response = np.zeros((n_cf, n_bins))
    for cf_idx in range(n_cf):
        padded = np.concatenate([np.full(K - 1, signal[cf_idx, 0]), signal[cf_idx]])
        on_response[cf_idx] = np.convolve(padded, kernel_on, mode='valid')[:n_bins]
        off_response[cf_idx] = np.convolve(padded, kernel_off, mode='valid')[:n_bins]
    if rectify:
        on_response = np.maximum(on_response, 0.0)
        off_response = np.maximum(off_response, 0.0)
    return on_response, off_response, K


def process_sweep(
    population_psth: np.ndarray,
    cf_hz: np.ndarray,
    dt_ms: float,
    alpha: float = ALPHA,
    population_scale: Optional[float] = None,
    w: float = ADAPTRANS_W,
    K: Optional[int] = ADAPTRANS_K,
    rectify: bool = ADAPTRANS_RECTIFY,
    tau_ms: float = ADAPTRANS_TAU_MS,
    tau_ms_off: Optional[float] = ADAPTRANS_TAU_MS_OFF,
) -> dict:
    """Power-law sharpening, then AdapTrans ON/OFF, on one continuous population PSTH.

    Parameters
    ----------
    population_psth : np.ndarray, shape (n_cf, n_bins)
        Raw population rate (spikes/s).
    cf_hz : np.ndarray, shape (n_cf,)
    dt_ms : float
        Bin width in ms (1.0 for the 1 kHz PSTH).
    population_scale : float, optional
        Pool rescale factor from compute_population_scale; None self-normalizes.

    Returns
    -------
    dict with sharpened, on_response, off_response (each (n_cf, n_bins)) and K (int,
    the AdapTrans kernel length used).
    """
    assert population_psth.ndim == 2 and population_psth.shape[0] == len(cf_hz)
    sharpened = apply_powerlaw_population(population_psth, alpha, population_scale)
    on_response, off_response, K = apply_adaptrans_continuous(
        sharpened, dt_ms, w, K, rectify, tau_ms, tau_ms_off)
    assert on_response.shape == off_response.shape == sharpened.shape
    return {"sharpened": sharpened, "on_response": on_response,
            "off_response": off_response, "K": K}


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_sweep_stages(result: dict, stages: dict, out_path: Path,
                      t_end_s: float = PLOT_SECONDS) -> None:
    """Raw, sharpened, ON and OFF (n_cf, n_bins) as CF (log) x time images, first t_end_s."""
    time_s, cf_hz = result["time_axis"], result["cf_list"]
    keep = time_s <= t_end_s
    panels = (("population PSTH (raw)", result["population_rate_psth"]),
              ("after power-law sharpening", stages["sharpened"]),
              ("AdapTrans ON", stages["on_response"]),
              ("AdapTrans OFF", stages["off_response"]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11), sharex=True,
                             constrained_layout=True)
    for ax, (title, array) in zip(axes, panels):
        mesh = ax.pcolormesh(time_s[keep], cf_hz, array[:, keep], shading="nearest",
                             cmap="magma", rasterized=True)
        ax.set_yscale("log")
        ax.set_ylabel("CF (Hz, log)")
        ax.set_title(title, fontsize=10, loc="left")
        fig.colorbar(mesh, ax=ax, pad=0.01)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(result["soundfileid"], fontsize=11)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── Orchestration ─────────────────────────────────────────────────────────────
def _format_param_suffix(alpha: float, tau_ms: float, w: float,
                         tau_ms_off: Optional[float] = None) -> str:
    """e.g. ``_alpha2p00_tau100_w0p80`` (``_tauon100_tauoff200`` if tau_ms_off differs)."""
    if tau_ms_off is not None and tau_ms_off != tau_ms:
        tau_str = f"tauon{tau_ms:03.0f}_tauoff{tau_ms_off:03.0f}"
    else:
        tau_str = f"tau{tau_ms:03.0f}"
    return f"_alpha{alpha:.2f}_{tau_str}_w{w:.2f}".replace(".", "p")


def run_pipeline(
    results_dir: Path,
    output_dir: Path,
    alpha: float = ALPHA,
    w: float = ADAPTRANS_W,
    K: Optional[int] = ADAPTRANS_K,
    rectify: bool = ADAPTRANS_RECTIFY,
    tau_ms: float = ADAPTRANS_TAU_MS,
    tau_ms_off: Optional[float] = ADAPTRANS_TAU_MS_OFF,
    exclude_n_cycles: tuple = (),
    save_plots: bool = True,
) -> dict:
    """Run sharpening + AdapTrans on every combined .npz in results_dir.

    Saves one .npz per stimulus in output_dir (sharpened, on_response, off_response,
    cf_list, time_axis, soundfileid, metadata and all parameters) and, optionally, one
    stage plot per stimulus in output_dir/plots.

    Returns
    -------
    dict {soundfileid: Path of the saved .npz}
    """
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    npz_files = select_files(sorted(results_dir.glob("*.npz")), exclude_n_cycles)
    if not npz_files:
        raise FileNotFoundError(f"No sweep .npz files in {results_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if save_plots:
        plot_dir.mkdir(exist_ok=True)
    logger.info("%d sweep file(s); alpha=%.2f w=%.2f tau_on=%.1f ms tau_off=%s rectify=%s",
                len(npz_files), alpha, w, tau_ms, tau_ms_off, rectify)

    # Phase 0: one rescale factor over the pre-power-law pool of all stimuli
    population_scale = compute_population_scale(iter_population_psths(npz_files), alpha)
    logger.info("Population power-law scale (alpha=%.2f): %.6e", alpha, population_scale)

    suffix = _format_param_suffix(alpha, tau_ms, w, tau_ms_off)
    written = {}
    for path in npz_files:
        result = load_sweep_result(path)
        dt_ms = 1000.0 * float(np.median(np.diff(result["time_axis"])))
        stages = process_sweep(result["population_rate_psth"], result["cf_list"], dt_ms,
                               alpha, population_scale, w, K, rectify, tau_ms, tau_ms_off)
        out_path = output_dir / f"{result['soundfileid']}{suffix}.npz"
        np.savez_compressed(
            out_path, soundfileid=result["soundfileid"], metadata=result["metadata"],
            cf_list=result["cf_list"], time_axis=result["time_axis"],
            sharpened=stages["sharpened"], on_response=stages["on_response"],
            off_response=stages["off_response"], alpha=alpha,
            population_scale=population_scale, w=w, K=stages["K"], rectify=rectify,
            tau_ms=tau_ms, tau_ms_off=tau_ms if tau_ms_off is None else tau_ms_off,
            dt_ms=dt_ms)
        if save_plots:
            plot_sweep_stages(result, stages, plot_dir / f"{out_path.stem}.png")
        written[result["soundfileid"]] = out_path
        logger.info("Saved %s", out_path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Folder of combined CochleaWavSimulation .npz files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: ./output/<results parent>_sweeps_adaptrans.")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--w", type=float, default=ADAPTRANS_W)
    parser.add_argument("--tau-ms", type=float, default=ADAPTRANS_TAU_MS)
    parser.add_argument("--tau-ms-off", type=float, default=ADAPTRANS_TAU_MS_OFF)
    parser.add_argument("--no-rectify", action="store_true")
    parser.add_argument("--exclude-n-cycles", type=int, nargs="*", default=[],
                        help="Skip stimuli with these cycle counts (e.g. 398).")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path("output") / f"{results_dir.parent.name}_sweeps_adaptrans")
    LoggingConfigurator(output_dir=output_dir, log_filename="pipeline_sweeps_adaptrans.log",
                        file_level=logging.DEBUG, console_level=logging.INFO).setup()
    run_pipeline(results_dir, output_dir, alpha=args.alpha, w=args.w,
                 rectify=not args.no_rectify, tau_ms=args.tau_ms,
                 tau_ms_off=args.tau_ms_off,
                 exclude_n_cycles=tuple(args.exclude_n_cycles),
                 save_plots=not args.no_plots)


if __name__ == "__main__":
    main()
