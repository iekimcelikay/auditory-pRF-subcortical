# full_pipeline_with_adaptrans.py
#
# PARAMETERS TO FIT = {cf_index, stimulus_id, sharpening_factor, preferred_duration, sigma_duration}
# Parameter to fit = Theta
# cf_index = k
# stimulus_id = s,
# sharpening_factor = alpha,
# preferred_duration = tau0,
# sigma_duration = sigma_tau0


import numpy as np
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Package level imports
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from auditory_prf.utils.cochlea_loader_functions import load_cochlea_results, organize_for_eachtone_allCFs, resolve_results_dir
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_cf_timecourse, load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_power_normalize, apply_powerlaw_cf, apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id

# Duration (scalar)
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar

# AdapTrans ON + OFF filters
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train, apply_adaptrans

# HRF convolution
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, convolve_hrf, hrf_summary, SUBCORTICAL_PARAMS

# ---- FUNCTIONS THAT ARE USED:
# _____________________________________________________________________________
# ---- 1 & 2 Load Cochlea Results, Extract one time course
# script: load_extract_cf_timecourse.py
#
# get_cf_timecourse(data: dict, cf) -> tuple[np.ndarray, int, float]
# load_cf_timecourse(npz_path: Path, cf) -> tuple[np.ndarray, np.ndarray, int, float, str]
# _____________________________________________________________________________
# ---- 3. Apply Sharpening with alpha (Lateral Inhibition stage)
# script: powerlaw_function.py
#
# apply_power_normalize(exp_name, results_dir, alpha, out_dir=None)
# _____________________________________________________________________________
# ---- 4. Tone-ON chunk timecourse
# script: chunk_timecourse.py
#
# parse_tone_timing(seq_id)  ->  (tone_dur_ms, isi_ms)
# compute_tone_onsets_offsets(tone_dur_ms, isi_ms, total_dur_ms)
#     -> (onsets_ms, offsets_ms)   — all in ms
# chunk_timecourse(timecourse, time_axis_s, tone_dur_ms, isi_ms, total_dur_ms,
#                 margin_ms=50.0)  ->  ChunkResult
#   ChunkResult.chunks         : List[ndarray]  — one array per tone
#   ChunkResult.axes_abs_ms    : List[ndarray]  — absolute time axis in ms
#   ChunkResult.axes_rel_ms    : List[ndarray]  — time axis in ms from each tone onset
#   ChunkResult.onsets_ms      : ndarray (num_tones,)
#   ChunkResult.offsets_ms     : ndarray (num_tones,)
#   ChunkResult.dt_ms          : float — bin width in ms (inferred from time_axis_s)
# _____________________________________________________________________________
# ---- 5. Gaussian duration filter (stimulus duration is SCALAR)
# script: duration_models.py
#
# apply_duration_gaussian_scalar(mean_rate_on: float, stim_dur: float,
#                                    pref_dur: float, sigma_dur: float) -> float
# _____________________________________________________________________________
# ++++ PIPELINE ++++
# _____________________________________________________________________________
#===== 1. LOAD COCHLEA RESULTS
# ── defaults ────────────────────────────────────────────────────────────────

#===== 2. EXTRACT ONE TIME COURSE
# get_cf_timecourse()
#===== 3. APPLY SHARPENING (LATERAL INHIBITION) WITH ALPHA
# apply_power_normalize()

# sharpened = apply_powerlaw_cf(timecourse, alpha)
#
#
#===== 4. CUT TO CHUNKS FOR TONE-ON, TAKE THE AVERAGE FIRING RATE = 1 VALUE PER TONE
# Chunk the power-normalized 1-D timecourse into tone-on windows (+ 50 ms margin).
# dt_ms is inferred from time_axis_s automatically.
#
# tone_dur_ms, isi_ms = parse_tone_timing(seq_id)
# total_dur_ms = time_axis_s[-1] * 1000 + (time_axis_s[1] - time_axis_s[0]) * 1000
# result = chunk_timecourse(sharpened, time_axis_s, tone_dur_ms, isi_ms, total_dur_ms)
# # result.onsets_ms, result.offsets_ms  — tone timing in ms
# # result.chunks[i]                     — firing rates in the i-th tone window
# # mean_rate_on = [np.mean(c) for c in result.chunks]   — one scalar per tone
# # tone_dur_ms  = tone_dur_ms                           — same for every tone (pure-tone sequence)
#===== 5. MULTIPLY BY DURATION SELECTIVE GAUSSIAN
# Multiply mean_rate_on by gaussian_duration(tone_dur, pref_dur, sigma_dur)
# add an `apply_duration_gaussian` function:
#   inputs:
#   - mean_rate_on
#   - tone_dur
#   - pref_dur
#   - sigma_dur
#   returns:
#   - mean_rate_on * gaussian_duration(tone_dur, pref_dur, sigma_dur)
# apply_duration_gaussian_scalar()
# prf_response = apply_duration_gaussian_scalar(mean_rate_on, tone_dur, pref_dur, sigma_dur)



# Module-level logger -- inherits handlers set up by LoggingConfigurator
logger = logging.getLogger(__name__)

EXP_NAME = "dipc_test_250225_01"
DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")

def _save_fig(fig, plot_dir: Path, name: str):
    """Save figure to plot_dir and close it."""
    fig.savefig(plot_dir / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.debug("  Saved plot: %s", name)


def run_pipeline(
        exp_name: str = EXP_NAME,
        results_dir: Path = None,
        alpha: float = 2.0,
        pref_dur: float = 200.0,
        sigma_dur: float = 20.0,
        output_dir: Path = None,
        cf=10,
        w: float = 0.8,
        K: int = None,
        save_plots: bool = True,
        hrf_params: dict = None,
        tr_s: float = 1.0,
        apply_hrf: bool = True,
        apply_duration_gaussian: bool = True,
        apply_adaptrans: bool = True,
        rectify: bool = False,
):

    # --- Logging setup -----
    _output_dir = output_dir or Path(f"./output/{exp_name}")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="prf_pipeline_with_adaptrans.log",
        file_level=logging.DEBUG,
        console_level=logging.DEBUG,
        ).setup()

    # --- Plot directory
    if save_plots:
        plot_dir = _output_dir / "intermediate_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Intermediate plots → %s", plot_dir)

    # --- 1. Resolve paths
    _results_dir = resolve_results_dir(
        results_dir if results_dir is not None else DEFAULT_BASE_DIR
        )
    logger.info("Experiment        :%s", exp_name)
    logger.info("Results directory : %s", _results_dir)

    npz_files = sorted(_results_dir.glob("*.npz"))
    if not npz_files:
        logger.error("No .npz files found in %s", _results_dir)
        sys.exit(1)
    logger.info("Found %d .npz file(s)", len(npz_files))

    # --- Build HRF kernel (once, before the file loop) ----------------------
    signal_dt_s = 1e-3   # AdapTrans output is always 1 ms
    if apply_hrf:
        _hrf_params = hrf_params if hrf_params is not None else SUBCORTICAL_PARAMS
        hrf_kernel, hrf_t = build_hrf_kernel(**_hrf_params, dt=signal_dt_s, duration=32.0)
        hrf_summary(hrf_kernel, hrf_t, signal_dt_s, _hrf_params)
        logger.info("HRF kernel built: %d samples | TR=%.3f s", len(hrf_kernel), tr_s)

    # Per-experiment, per-sequence results: {exp_name: {seq_id: {...}}}
    all_results: dict[str, dict[str, dict]] = {exp_name: {}}

    # --- 2-5. Per-file loop --------------------------------------------------
    for i, npz_path in enumerate(npz_files, 1):
        logger.info("[%d/%d] Processing: %s", i, len(npz_files), npz_path.name)

        # 2. Load full population PSTH (all CFs × all time bins)
        population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(npz_path, cf)
        logger.debug("   CF index: %d | CF Hz: %.1f | seq_id: %s", cf_index, cf_hz, seq_id)
        dt_s = time_axis[1] - time_axis[0]
        total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
        logger.debug("   total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

        # 3. Apply power-law sharpening across all CFs, then extract target CF.
        # Mean across all cochlear channels (all CFs × all time bins) is preserved.
        sharpened_pop = apply_powerlaw_population(population_psth, alpha)
        sharpened = sharpened_pop[cf_index, :]
        logger.debug("  Sharpened timecourse shape: %s", sharpened.shape)

        # --- Plot: powerlaw sharpening (raw vs sharpened for target CF)
        if save_plots:
            raw_cf = population_psth[cf_index, :]
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(time_axis * 1000, raw_cf, linewidth=0.6, color='steelblue')
            axes[0].set_ylabel("Rate (spk/s)")
            axes[0].set_title(f"Raw CF timecourse — CF {cf_hz:.0f} Hz | {seq_id}")
            axes[1].plot(time_axis * 1000, sharpened, linewidth=0.6, color='darkorange')
            axes[1].set_ylabel("Rate (spk/s)")
            axes[1].set_xlabel("Time (ms)")
            axes[1].set_title(f"After power-law sharpening (α={alpha})")
            _save_fig(fig, plot_dir, f"01_powerlaw_{seq_id}.png")

        # 4. Chunk into tone-on windows
        result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id)
        n_tones = len(result["chunks"])
        logger.debug("  Tone dur: %.1f ms | ISI: %.1f ms | n_tones %d",
                     tone_dur_ms, isi_ms, n_tones)


        # get the mean rates of tone-on chunks
        mean_rates_on = [np.mean(c) for c in result["chunks"]]
        logger.debug("  Mean rates per tone: %s", [f"{m:.2f}" for m in mean_rates_on])

        # --- Plot: mean firing rate per tone-ON chunk
        if save_plots:
            fig, ax = plt.subplots(figsize=(10, 4))
            tone_indices = np.arange(1, n_tones + 1)
            ax.bar(tone_indices, mean_rates_on, color='teal', edgecolor='k', linewidth=0.5)
            ax.set_xlabel("Tone number")
            ax.set_ylabel("Mean rate (spk/s)")
            ax.set_title(f"Tone-ON mean rates — {seq_id} | dur={tone_dur_ms:.0f}ms, ISI={isi_ms:.0f}ms")
            _save_fig(fig, plot_dir, f"02_chunk_mean_rates_{seq_id}.png")

        # 5. Apply duration Gaussian (Scalar)
        prf_responses = [
            apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur, sigma_dur)
            for m in mean_rates_on
            ]
        logger.debug("  pRF responses(first 3): %s", prf_responses[:3])

        # --- Plot: duration-weighted pRF responses vs raw mean rates
        if save_plots:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(tone_indices - 0.2, mean_rates_on, width=0.4,
                   label="Raw mean rate", color='teal', alpha=0.6)
            ax.bar(tone_indices + 0.2, prf_responses, width=0.4,
                   label=f"Duration-weighted (τ₀={pref_dur:.0f}, σ={sigma_dur:.0f})",
                   color='salmon', alpha=0.8)
            ax.set_xlabel("Tone number")
            ax.set_ylabel("Rate (spk/s)")
            ax.set_title(f"Duration Gaussian weighting — {seq_id}")
            ax.legend()
            _save_fig(fig, plot_dir, f"03_duration_weighted_{seq_id}.png")

        # 6. Build full multi-tone boxcar train
        amplitudes = prf_responses if apply_duration_gaussian else mean_rates_on
        train = build_prf_boxcar_train(
            amplitudes, result["onsets_ms"], result["offsets_ms"],
            total_dur_ms, dt_ms=1.0,
        )
        n_1ms = len(train)

        # 7. Apply AdapTrans to the full train (all tones at once, preserving carry-over)
        if apply_adaptrans:
            on_off = apply_adaptrans(
                train[np.newaxis, :],
                CFs_Hz=np.array([cf_hz]),
                dt_ms=1.0,
                w=w,
                K=K,
                pad_value=0.0,
                rectify=rectify,
            )
            on_response  = on_off[0, 0, :]
            off_response = on_off[1, 0, :]
        else:
            on_response  = train
            off_response = np.zeros(n_1ms)

        logger.debug("  ON  response shape: %s | min: %.4e | max: %.4e",
                     on_response.shape, on_response.min(), on_response.max())
        logger.debug("  OFF response shape: %s | min: %.4e | max: %.4e",
                     off_response.shape, off_response.min(), off_response.max())

        # 8. HRF convolution → BOLD
        if apply_hrf:
            bold_on  = convolve_hrf(on_response,  hrf_kernel, signal_dt=signal_dt_s,
                                    kernel_dt=signal_dt_s, output_dt=tr_s)
            bold_off = convolve_hrf(off_response, hrf_kernel, signal_dt=signal_dt_s,
                                    kernel_dt=signal_dt_s, output_dt=tr_s)
            bold_combined = bold_on + bold_off
            logger.debug("  BOLD shape: on=%s  off=%s  combined=%s",
                         bold_on.shape, bold_off.shape, bold_combined.shape)
        else:
            bold_on = bold_off = bold_combined = None

        # --- Plot: AdapTrans ON/OFF responses + boxcar train
        if save_plots and apply_adaptrans:
            time_1ms = np.arange(n_1ms)
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(time_1ms, train, linewidth=0.6, color='gray')
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title(f"Boxcar train (pRF-weighted) — {seq_id}")
            axes[1].plot(time_1ms, on_response, linewidth=0.6, color='crimson')
            axes[1].set_ylabel("ON response")
            axes[1].set_title(f"AdapTrans ON (w={w}, CF={cf_hz:.0f} Hz)")
            axes[2].plot(time_1ms, off_response, linewidth=0.6, color='royalblue')
            axes[2].set_ylabel("OFF response")
            axes[2].set_xlabel("Time (ms)")
            axes[2].set_title("AdapTrans OFF")
            _save_fig(fig, plot_dir, f"04_adaptrans_{seq_id}.png")

        # --- Plot: BOLD timeseries (on, off, combined)
        if save_plots and apply_hrf:
            t_tr = np.arange(len(bold_combined)) * tr_s
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(t_tr, bold_on, 'o-', ms=4, color='crimson')
            axes[0].set_ylabel("BOLD (a.u.)")
            axes[0].set_title(f"BOLD ON — {seq_id} | CF={cf_hz:.0f} Hz | TR={tr_s:.2f}s")
            axes[1].plot(t_tr, bold_off, 'o-', ms=4, color='royalblue')
            axes[1].set_ylabel("BOLD (a.u.)")
            axes[1].set_title("BOLD OFF")
            axes[2].plot(t_tr, bold_combined, 'o-', ms=4, color='forestgreen')
            axes[2].set_ylabel("BOLD (a.u.)")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_title("BOLD combined (ON + OFF)")
            _save_fig(fig, plot_dir, f"05_bold_{seq_id}.png")

        # Store this sequence's results under (exp_name, seq_id).
        # Warn (don't overwrite silently) if a duplicate seq_id appears.
        if seq_id in all_results[exp_name]:
            logger.warning("Duplicate seq_id '%s' under experiment '%s' — overwriting.",
                           seq_id, exp_name)
        all_results[exp_name][seq_id] = {
            "cf_index":               cf_index,
            "cf_hz":                  cf_hz,
            "prf_responses":          prf_responses,
            "on_response":            on_response,
            "off_response":           off_response,
            "train":                  train,
            "bold_on":                bold_on,
            "bold_off":               bold_off,
            "bold_combined":          bold_combined,
            "apply_duration_gaussian": apply_duration_gaussian,
            "apply_adaptrans":         apply_adaptrans,
            "rectify":                 rectify,
        }

    logger.info("Pipeline complete. %d sequence(s) processed for '%s'.",
                len(all_results[exp_name]), exp_name)
    return all_results


if __name__ == "__main__":
    run_pipeline()
