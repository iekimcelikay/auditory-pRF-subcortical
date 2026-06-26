"""full_pipeline_toneclouds_adaptrans.py
==========================================
pRF pipeline for tone-cloud (Gaussian filterbank) stimuli, WITH AdapTrans.

Same Phase 1 / Phase 2 run-assembly architecture as
full_pipeline_toneclouds_notemporal.py, but:
  - AdapTrans ON/OFF filtering is applied to the assembled run-level boxcar
    train (apply_adaptrans_flag=True by default), so carry-over adaptation
    between back-to-back trials within a run is modelled.
  - w, K, rectify, rho are exposed as run_pipeline() kwargs.
  - Intermediate diagnostic plots are saved (ported from
    full_pipeline_with_adaptrans.py):
      Phase 1 (per sequence): powerlaw sharpening, tone-ON mean rates
      Phase 2 (per run):      assembled boxcar + AdapTrans ON/OFF,
                               BOLD ON/OFF/combined

Retained stages:
  1. Load cochlea PSTH results
  2. Power-law sharpening (alpha)
  3. Chunk into tone-ON windows -> mean rate per tone
  3a. Duration Gaussian filter (optional: pref_dur / sigma_dur in ms)
  4. Build boxcar train (amplitude = Gaussian-weighted mean rate, or raw mean rate)
  5. Assemble N runs from run design
  6. AdapTrans ON/OFF on the assembled run-level train
  7. HRF convolution -> BOLD
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.run_assembly import (
    generate_run_design, assemble_run_bold, apply_run_noise, parse_noise_seed_arg,
)
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from prf_models.pm_noise import PmNoise

logger = logging.getLogger(__name__)


# ── plotting helpers ──────────────────────────────────────────────────────────
def _save_fig(fig, plot_dir: Path, name: str):
    """Save figure to plot_dir and close it."""
    fig.savefig(plot_dir / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.debug("  Saved plot: %s", name)


def _plot_sharpening(raw_rates, sharpened_rates, alpha, seq_id, plot_dir):
    """Phase 1 plot: per-tone mean rates before and after power-law sharpening."""
    n_tones = len(raw_rates)
    tone_indices = np.arange(1, n_tones + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(tone_indices, raw_rates, color='steelblue', edgecolor='k', linewidth=0.5)
    axes[0].set_ylabel("Mean rate (spk/s)")
    axes[0].set_title(f"Raw mean rates per tone — {seq_id}")
    axes[1].bar(tone_indices, sharpened_rates, color='darkorange', edgecolor='k', linewidth=0.5)
    axes[1].set_ylabel("Mean rate (spk/s)")
    axes[1].set_xlabel("Tone number")
    axes[1].set_title(f"After power-law sharpening (α={alpha})")
    _save_fig(fig, plot_dir, f"01_sharpening_{seq_id}.png")


def _plot_chunk_mean_rates(mean_rates_on, seq_id, plot_dir):
    """Phase 1 plot: sharpened mean firing rate per tone-ON chunk."""
    n_tones = len(mean_rates_on)
    tone_indices = np.arange(1, n_tones + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(tone_indices, mean_rates_on, color='teal', edgecolor='k', linewidth=0.5)
    ax.set_xlabel("Tone number")
    ax.set_ylabel("Mean rate (spk/s)")
    ax.set_title(f"Tone-ON mean rates (sharpened) — {seq_id}")
    _save_fig(fig, plot_dir, f"02_chunk_mean_rates_{seq_id}.png")


def _plot_run_adaptrans(run_idx, result, cf_hz, w, plot_dir, combo_suffix=""):
    """Phase 2 plot: assembled boxcar train + AdapTrans ON/OFF for one run."""
    n_1ms = len(result["full_train"])
    time_1ms = np.arange(n_1ms)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time_1ms, result["full_train"], linewidth=0.6, color='gray')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Assembled boxcar train — run {run_idx + 1:02d}{combo_suffix}")
    axes[1].plot(time_1ms, result["on_response"], linewidth=0.6, color='crimson')
    axes[1].set_ylabel("ON response")
    axes[1].set_title(f"AdapTrans ON (w={w}, CF={cf_hz:.0f} Hz)")
    axes[2].plot(time_1ms, result["off_response"], linewidth=0.6, color='royalblue')
    axes[2].set_ylabel("OFF response")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("AdapTrans OFF")
    _save_fig(fig, plot_dir, f"03_adaptrans_run{run_idx + 1:02d}{combo_suffix}.png")


def _plot_run_bold(run_idx, result, cf_hz, tr_s, rho, plot_dir, combo_suffix=""):
    """Phase 2 plot: BOLD ON / OFF / combined for one run."""
    t_tr = result["t_tr"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_tr, result["bold_on"], 'o-', ms=4, color='crimson')
    axes[0].set_ylabel("BOLD (a.u.)")
    axes[0].set_title(f"BOLD ON — run {run_idx + 1:02d}{combo_suffix} | CF={cf_hz:.0f} Hz | TR={tr_s:.2f}s")
    axes[1].plot(t_tr, result["bold_off"], 'o-', ms=4, color='royalblue')
    axes[1].set_ylabel("BOLD (a.u.)")
    axes[1].set_title("BOLD OFF")
    axes[2].plot(t_tr, result["bold_combined"], 'o-', ms=4, color='forestgreen')
    axes[2].set_ylabel("BOLD (a.u.)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"BOLD combined (rho * ON + OFF, rho={rho})")
    _save_fig(fig, plot_dir, f"04_bold_run{run_idx + 1:02d}{combo_suffix}.png")


def _save_run_plots(run_idx, result, cf_hz, w, rho, tr_s, plot_dir, combo_suffix=""):
    """Save both Phase 2 (per-run) diagnostic plots."""
    _plot_run_adaptrans(run_idx, result, cf_hz, w, plot_dir, combo_suffix)
    _plot_run_bold(run_idx, result, cf_hz, tr_s, rho, plot_dir, combo_suffix)


def _strip_signal_arrays(result: dict) -> dict:
    """Drop the 1 ms-resolution arrays (full_train, on/off_response).

    Only the much smaller TR-resolution arrays (bold_on, bold_off,
    bold_combined, t_tr) need to travel back through multiprocessing IPC.
    """
    return {k: v for k, v in result.items()
            if k not in ("full_train", "on_response", "off_response")}


def _format_combo_suffix(tau_ms: float, w: float, rho: float) -> str:
    """Build a filename/title suffix identifying one AdapTrans param combo.

    e.g. ``_tau050_w0p80_rho1p00``
    """
    tau_str = f"tau{tau_ms:03.0f}"
    w_str   = f"w{w:.2f}".replace(".", "p")
    rho_str = f"rho{rho:.2f}".replace(".", "p")
    return f"_{tau_str}_{w_str}_{rho_str}"


# ── multiprocessing worker (module-level so it is picklable) ─────────────────
_worker: dict = {}


def _worker_init(per_seq, hrf_kernel, total_run_dur_s, cf_hz, tr_s, signal_dt_s, noise_models,
                  apply_adaptrans_flag, w, K, rectify, rho, tau_ms,
                  save_plots, plot_dir, combo_suffix):
    _worker["per_seq"]              = per_seq
    _worker["hrf_kernel"]           = hrf_kernel
    _worker["total_run_dur_s"]      = total_run_dur_s
    _worker["cf_hz"]                = cf_hz
    _worker["tr_s"]                 = tr_s
    _worker["signal_dt_s"]          = signal_dt_s
    _worker["noise_models"]         = noise_models
    _worker["apply_adaptrans_flag"] = apply_adaptrans_flag
    _worker["w"]                    = w
    _worker["K"]                    = K
    _worker["rectify"]              = rectify
    _worker["rho"]                  = rho
    _worker["tau_ms"]               = tau_ms
    _worker["save_plots"]           = save_plots
    _worker["plot_dir"]             = plot_dir
    _worker["combo_suffix"]         = combo_suffix


def _assemble_one(task: tuple) -> tuple:
    run_idx, run_design = task
    result = assemble_run_bold(
        per_seq              = _worker["per_seq"],
        run_design           = run_design,
        total_run_dur_s      = _worker["total_run_dur_s"],
        hrf_kernel           = _worker["hrf_kernel"],
        cf_hz                = _worker["cf_hz"],
        tr_s                 = _worker["tr_s"],
        signal_dt_s          = _worker["signal_dt_s"],
        w                    = _worker["w"],
        K                    = _worker["K"],
        apply_adaptrans_flag = _worker["apply_adaptrans_flag"],
        rectify              = _worker["rectify"],
        rho                  = _worker["rho"],
        tau_ms               = _worker["tau_ms"],
    )
    bold_noisy_by_level = {
        level: apply_run_noise(result["bold_combined"], noise_model, run_idx, _worker["tr_s"])
        for level, noise_model in _worker["noise_models"].items()
    }
    if _worker["save_plots"]:
        _save_run_plots(run_idx, result, _worker["cf_hz"], _worker["w"], _worker["rho"],
                        _worker["tr_s"], _worker["plot_dir"], _worker["combo_suffix"])
    return run_idx, run_design, _strip_signal_arrays(result), bold_noisy_by_level


# ── tone-cloud seq_id scheme ───────────────────────────────────────────────────
BAND_CENTERS_HZ   = (572, 885, 1322)   # ERB-spaced Gaussian filterbank centers (450-1600 Hz, N=3)
TOTAL_SEQ_DUR_S   = 20.0               # WAV sequence duration (seconds)
STIMULUS_SAMPLE_RATE = 100_000         # sample rate used during WAV generation
TC_SILENCE_SEQ_ID = "tonecloud00_dur0ms_isi0ms"


def _make_tonecloud_seq_id_fn(
    band_centers_hz: tuple,
    total_seq_dur_s: float,
    sample_rate: int,
):
    """Return a callable ``(dur_ms, isi_ms, g_idx) -> seq_id str``.

    Reproduces the filename produced by save_tone_clouds_gaussian_prf.py:
        tonecloud{g:02d}_fc{hz}hz_dur{D}ms_isi{I}ms_numtones{N}

    numtones is computed with the same sample-accurate floor-division formula as
    calculate_num_tones() in stimuli/sample_tone_cloud_freqs_gaussian.py so that
    keys match the stored soundfileid exactly. Pass float dur_ms values (not
    rounded integers) to avoid off-by-one rounding errors for long durations.

    ``g_idx=None`` returns TC_SILENCE_SEQ_ID (null / silence trial).
    """
    total_samples = int(total_seq_dur_s * sample_rate)

    def _fn(dur_ms, isi_ms, g_idx):
        if g_idx is None:
            return TC_SILENCE_SEQ_ID
        center_hz   = band_centers_hz[g_idx]
        tone_samples = int(dur_ms / 1000.0 * sample_rate)
        isi_samples  = int(isi_ms / 1000.0 * sample_rate)
        numtones     = total_samples // (tone_samples + isi_samples)
        return (
            f"tonecloud{g_idx + 1:02d}"
            f"_fc{int(round(center_hz))}hz"
            f"_dur{int(round(dur_ms))}ms"
            f"_isi{int(round(isi_ms))}ms"
            f"_numtones{numtones}"
        )
    return _fn


# ── experiment defaults ───────────────────────────────────────────────────────
EXP_NAME         = "toneclouds_gaussianprf"
DEFAULT_BASE_DIR  = Path(f"./models_output/{EXP_NAME}")

# ── run design defaults ───────────────────────────────────────────────────────
# Float durations from find_closest_durations() in find_optimal_durations.py.
# Must be floats (not rounded ints) so that numtones computation matches
# calculate_num_tones() used during WAV generation.
TONE_ON_MS       = (34.89, 44.76, 60.14, 75.38, 100.44, 149.72, 247.58, 496.43)

ISI_MS           = (75,) * len(TONE_ON_MS)
NULL_FRACTION    = 0.25
TRIAL_DURATION_S  = 20.0
OPENING_BLANK_S   = 10.0
CLOSING_BLANK_S   = 10.0
ITI_RANGE_S       = 0
N_RUNS            = 24
BASE_SEED         = 42

# ── AdapTrans / BOLD defaults ──────────────────────────────────────────────────
ADAPTRANS_W       = 0.8
ADAPTRANS_K       = None    # auto-set (3x longest CF time constant)
ADAPTRANS_RECTIFY = True
BOLD_RHO          = 1.0
ADAPTRANS_TAU_MS  = 100.0   # free parameter, no CF-tau relationship


def run_pipeline(
        exp_name: str = EXP_NAME,
        results_dir: Optional[Path] = None,
        alpha: float = 2.0,
        pref_dur: Optional[float] = 75,   # ms; None = skip duration Gaussian
        sigma_dur: float = 50.0,             # ms
        cf=10,
        output_dir: Optional[Path] = None,
        # tone-cloud filterbank
        band_centers_hz: tuple = BAND_CENTERS_HZ,
        total_seq_dur_s: float = TOTAL_SEQ_DUR_S,
        stimulus_sample_rate: int = STIMULUS_SAMPLE_RATE,
        # run design
        tone_on_ms: tuple = TONE_ON_MS,
        isi_ms: tuple = ISI_MS,
        null_fraction: float = NULL_FRACTION,
        trial_duration_s: float = TRIAL_DURATION_S,
        opening_blank_s: float = OPENING_BLANK_S,
        closing_blank_s: float = CLOSING_BLANK_S,
        iti_range_s: Union[float, tuple] = ITI_RANGE_S,
        n_runs: int = N_RUNS,
        base_seed: int = BASE_SEED,
        run_designs: Optional[list] = None,
        total_run_dur_s: float = 720.0,
        # HRF
        hrf_params: Optional[dict] = None,
        tr_s: float = 1.6,
        # AdapTrans
        apply_adaptrans_flag: bool = True,
        w: float = ADAPTRANS_W,
        K: Optional[int] = ADAPTRANS_K,
        rectify: bool = ADAPTRANS_RECTIFY,
        rho: float = BOLD_RHO,
        tau_ms: float = ADAPTRANS_TAU_MS,
        param_grid: Optional[list[dict]] = None,
        # plotting
        save_plots: bool = True,
        # parallelism
        n_workers: int = 1,
        # noise
        noise_models: Optional[dict[str, PmNoise]] = None,
):
    noise_models = noise_models or {}
    # param_grid: list of {"tau_ms", "w", "rho"} combos to sweep. Phase 1 runs
    # once regardless; Phase 2 (AdapTrans + HRF) and the saved npz run once per
    # combo, with (tau_ms, w, rho) encoded in the filename (_format_combo_suffix).
    if param_grid is None:
        param_grid = [{"tau_ms": tau_ms, "w": w, "rho": rho}]
    _output_dir = output_dir or Path(f"./output/{exp_name}_toneclouds_adaptrans")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="pipeline_toneclouds_adaptrans.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    ).setup()

    plot_dir = None
    if save_plots:
        plot_dir = _output_dir / "intermediate_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Intermediate plots → %s", plot_dir)

    _results_dir = (results_dir if results_dir is not None else DEFAULT_BASE_DIR)
    _results_dir = Path(_results_dir).expanduser().resolve()
    logger.info("Experiment       : %s", exp_name)
    logger.info("Results dir      : %s", _results_dir)
    logger.info("Band centres (Hz): %s", band_centers_hz)
    logger.info("AdapTrans        : apply=%s, K=%s, rectify=%s",
                 apply_adaptrans_flag, K, rectify)

    npz_files = sorted(_results_dir.glob("wav*/**/*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {_results_dir}/wav*/. "
            "Run the cochlear simulation first and check results_dir."
        )
    logger.info("Found %d NPZ file(s)", len(npz_files))

    logger.info("Param grid       : %d combo(s) — %s", len(param_grid), param_grid)
    logger.info("Noise levels     : %s", list(noise_models.keys()) or "none")

    # ── HRF kernel (built once) ───────────────────────────────────────────────
    signal_dt_s = 1e-3
    _hrf_params = hrf_params if hrf_params is not None else SUBCORTICAL_PARAMS
    hrf_kernel, _ = build_hrf_kernel(**_hrf_params, dt=signal_dt_s, duration=32.0)
    logger.info("HRF kernel: %d samples | TR=%.3f s", len(hrf_kernel), tr_s)

    # ── build run design params ───────────────────────────────────────────────
    n_gaussians = len(band_centers_hz)
    stimuli     = [(ton, isi, g_idx)
                   for ton, isi in zip(tone_on_ms, isi_ms)
                   for g_idx in range(n_gaussians)]
    n_null      = int(np.floor(len(stimuli) * null_fraction / (1 - null_fraction)))
    base_trials = stimuli + [(0, 0, None)] * n_null

    seq_id_fn = _make_tonecloud_seq_id_fn(
        band_centers_hz, total_seq_dur_s, stimulus_sample_rate
    )
    logger.info("Base trials: %d active + %d null = %d total",
                len(stimuli), n_null, len(base_trials))

    # ── Phase 1: per-sequence processing ──────────────────────────────────────
    per_seq: dict = {}

    for i, npz_path in enumerate(npz_files, 1):
        logger.info("[%d/%d] %s", i, len(npz_files), npz_path.name)

        population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(
            npz_path, cf
        )
        dt_s         = time_axis[1] - time_axis[0]
        total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
        # . CHUNK INTO TONE-ON WINDOWS → MEAN RATES → POWER LAW SHARPENING
        cf_tc_raw = population_psth[cf_index, :]

        if seq_id == TC_SILENCE_SEQ_ID:
            spont_rate = float(np.mean(cf_tc_raw))
            n_samples  = int(round(total_dur_ms))
            train      = np.full(n_samples, spont_rate)
            logger.debug("  seq_id=%s | CF=%.0f Hz | spont_rate=%.2f sp/s | train len=%d",
                         seq_id, cf_hz, spont_rate, len(train))
        else:
            result, tone_dur_ms, _ = chunk_from_id(cf_tc_raw, time_axis, seq_id)
            raw_mean_rates = np.array([np.mean(c) for c in result["chunks"]])
            mean_rates_on  = apply_powerlaw_population(raw_mean_rates, alpha)

            if save_plots:
                _plot_sharpening(raw_mean_rates, mean_rates_on, alpha, seq_id, plot_dir)
                _plot_chunk_mean_rates(mean_rates_on, seq_id, plot_dir)
        # . (OPTIONAL) Gaussian duration filter
            if pref_dur is not None:
                amplitudes = [
                    apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur, sigma_dur)
                    for m in mean_rates_on
                ]
                logger.debug("  Duration Gaussian (pref_dur=%.0f ms, sigma=%.0f ms, tone_dur=%.0f ms) "
                             "| weight=%.4f | mean_amp: %.2f -> %.4f",
                             pref_dur, sigma_dur, tone_dur_ms,
                             amplitudes[0] / mean_rates_on[0] if mean_rates_on[0] else 0,
                             float(np.mean(mean_rates_on)), float(np.mean(amplitudes)))
            else:
                amplitudes = mean_rates_on
        # . Build boxcar train
            train = build_prf_boxcar_train(
                amplitudes,
                result["onsets_ms"],
                result["offsets_ms"],
                total_dur_ms,
                dt_ms=1.0,
            )
            logger.debug("  seq_id=%s | CF=%.0f Hz | %d tones | train len=%d",
                         seq_id, cf_hz, len(mean_rates_on), len(train))

        per_seq[seq_id] = {
            "train":    train,
            "cf_hz":    cf_hz,
            "cf_index": cf_index,
        }

    logger.info("Phase 1 complete — %d sequences processed.", len(per_seq))

    # ── Validate seq_id alignment ──────────────────────────────────────────────
    expected_seq_ids = {seq_id_fn(ton, isi, g_idx)
                        for ton, isi, g_idx in stimuli} | {TC_SILENCE_SEQ_ID}
    missing = expected_seq_ids - set(per_seq.keys())
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"Stimulus params do not match .npz files in results_dir.\n"
            f"{len(missing)} expected seq_ids not found in per_seq "
            f"(e.g. {sample}).\n"
            f"Check that BAND_CENTERS_HZ, TONE_ON_MS, and ISI_MS match "
            f"the WAV files used to generate the cochlear results."
        )

    # ── Phase 2: per-run assembly ──────────────────────────────────────────────
    cf_hz_used = next(iter(per_seq.values()))["cf_hz"]

    _designs = run_designs if run_designs is not None else [
        generate_run_design(
            base_trials, seq_id_fn,
            trial_duration_s=trial_duration_s,
            opening_blank_s=opening_blank_s,
            iti_range_s=iti_range_s,
            seed=base_seed + i,
        )
        for i in range(n_runs)
    ]
    tasks = list(enumerate(_designs))
    saver = ResultSaver(_output_dir)
    all_results: dict = {}

    for combo in param_grid:
        combo_tau_ms, combo_w, combo_rho = combo["tau_ms"], combo["w"], combo["rho"]
        combo_suffix = _format_combo_suffix(combo_tau_ms, combo_w, combo_rho)
        logger.info("Phase 2: assembling %d run(s) with %d worker(s) | combo%s",
                    len(_designs), n_workers, combo_suffix)

        all_runs: dict = {}

        if n_workers > 1:
            with Pool(
                processes=n_workers,
                initializer=_worker_init,
                initargs=(per_seq, hrf_kernel, total_run_dur_s,
                          cf_hz_used, tr_s, signal_dt_s, noise_models,
                          apply_adaptrans_flag, combo_w, K, rectify, combo_rho,
                          combo_tau_ms,
                          save_plots, plot_dir, combo_suffix),
            ) as pool:
                results = pool.map(_assemble_one, tasks)
        else:
            results = []
            for run_idx, run_design in tasks:
                result = assemble_run_bold(
                    per_seq=per_seq,
                    run_design=run_design,
                    total_run_dur_s=total_run_dur_s,
                    hrf_kernel=hrf_kernel,
                    cf_hz=cf_hz_used,
                    tr_s=tr_s,
                    signal_dt_s=signal_dt_s,
                    w=combo_w,
                    K=K,
                    apply_adaptrans_flag=apply_adaptrans_flag,
                    rectify=rectify,
                    rho=combo_rho,
                    tau_ms=combo_tau_ms,
                )
                bold_noisy_by_level = {
                    level: apply_run_noise(result["bold_combined"], noise_model, run_idx, tr_s)
                    for level, noise_model in noise_models.items()
                }
                if save_plots:
                    _save_run_plots(run_idx, result, cf_hz_used, combo_w, combo_rho, tr_s,
                                    plot_dir, combo_suffix)
                results.append((run_idx, run_design, _strip_signal_arrays(result), bold_noisy_by_level))

        for run_idx, run_design, result, bold_noisy_by_level in results:
            all_runs[f"run_{run_idx + 1:02d}"] = {
                "run_design":         run_design,
                "bold_combined":      result["bold_combined"],
                "bold_on":            result["bold_on"],
                "bold_off":           result["bold_off"],
                "bold_noisy_by_level": bold_noisy_by_level,
                "t_tr":               result["t_tr"],
                "seed":          base_seed + run_idx if run_designs is None else None,
            }
        if all_runs:
            logger.info("  BOLD shape: %s", next(iter(all_runs.values()))["bold_combined"].shape)

        # ── Save ──────────────────────────────────────────────────────────────
        save_dict = {
            "exp_name":             exp_name,
            "cf":                   cf,
            "alpha":                alpha,
            "pref_dur":             str(pref_dur),
            "sigma_dur":            sigma_dur,
            "tr_s":                 tr_s,
            "w":                    combo_w,
            "K":                    str(K),
            "rectify":              rectify,
            "rho":                  combo_rho,
            "apply_adaptrans_flag": apply_adaptrans_flag,
            "tau_ms":               str(combo_tau_ms),
            "cf_range_hz":          cf_range_hz,
            "tau_range_ms":         tau_range_ms,
            **{k: v["bold_combined"] for k, v in all_runs.items()},
            **{f"{k}_bold_on":  v["bold_on"]  for k, v in all_runs.items()},
            **{f"{k}_bold_off": v["bold_off"] for k, v in all_runs.items()},
        }
        for level, noise_model in noise_models.items():
            save_dict.update({
                f"{k}_noisy_{level}": v["bold_noisy_by_level"][level] for k, v in all_runs.items()
            })
            save_dict[f"noise_seed_{level}"] = str(noise_model.seed)
        pref_str = f"_pdur{int(round(pref_dur))}" if pref_dur is not None else ""
        saver.save_npz(
            save_dict,
            f"{exp_name}_toneclouds_adaptrans_cf{cf:03d}{pref_str}{combo_suffix}_bold.npz",
        )
        logger.info("Saved BOLD to %s", _output_dir)

        all_results[combo_suffix] = all_runs

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=int, default=0,
                        help="CF index (0-based). Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to cochlea NPZ results directory.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--pref_dur", type=float, default=None,
                        help="Preferred duration for Gaussian duration filter (ms). "
                             "None (default) = skip duration Gaussian.")
    parser.add_argument("--sigma_dur", type=float, default=50.0,
                        help="Duration tuning width for Gaussian filter (ms). Default: 50.")
    parser.add_argument("--w", type=float, default=ADAPTRANS_W,
                        help="AdapTrans adaptation weight.")
    parser.add_argument("--rho", type=float, default=BOLD_RHO,
                        help="ON-to-OFF BOLD weighting ratio "
                             "(bold_combined = rho * bold_on + bold_off).")
    parser.add_argument("--rectify", action=argparse.BooleanOptionalAction,
                        default=ADAPTRANS_RECTIFY,
                        help=f"Half-wave rectify AdapTrans ON/OFF output "
                             f"(default: {ADAPTRANS_RECTIFY}). Use --no-rectify to disable.")
    parser.add_argument("--no-adaptrans", action="store_true",
                        help="Disable AdapTrans (use the assembled boxcar train directly).")
    parser.add_argument("--tau_ms", type=float, default=ADAPTRANS_TAU_MS,
                        help=f"AdapTrans time constant (ms). Free parameter, no CF relationship. "
                             f"Default: {ADAPTRANS_TAU_MS}.")
    parser.add_argument("--tau_ms_sweep", type=float, nargs="+", default=None,
                        help="Sweep multiple AdapTrans tau_ms values (overrides --tau_ms). "
                             "E.g. --tau_ms_sweep 50 100 200. Phase 1 runs once; Phase 2 + "
                             "a separate npz are produced per (tau_ms, w, rho) combo.")
    parser.add_argument("--w_sweep", type=float, nargs="+", default=None,
                        help="Sweep multiple AdapTrans w values (overrides --w).")
    parser.add_argument("--rho_sweep", type=float, nargs="+", default=None,
                        help="Sweep multiple BOLD rho values (overrides --rho).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable intermediate diagnostic plots.")
    parser.add_argument("--noise_voxels", nargs="+", choices=["none", "low", "mid", "high"],
                        default=["none"],
                        help="BOLD noise preset(s) (PmNoise voxel level). Pass multiple to "
                             "compute several noisy variants from the same clean run "
                             "(e.g. --noise_voxels low mid high). 'none' = no noisy variant "
                             "(default).")
    parser.add_argument("--noise_seed", type=str, default="random",
                        help="PmNoise seed shared across all --noise_voxels levels "
                             "(each run's seed is still offset by run index): an integer "
                             "for reproducible noise, 'random' (default), or 'none'/'nonoise'.")
    args = parser.parse_args()

    _noise_models = {
        level: PmNoise(voxel=level, seed=parse_noise_seed_arg(args.noise_seed))
        for level in args.noise_voxels if level != "none"
    }

    # Build the (tau_ms, w, rho) sweep grid. Each *_sweep flag falls back to
    # the single-value --tau_ms/--w/--rho default when not given, so the
    # no-sweep case still produces a one-combo grid (with a filename suffix).
    import itertools

    _tau_ms_values = args.tau_ms_sweep if args.tau_ms_sweep else [args.tau_ms]
    _w_values      = args.w_sweep      if args.w_sweep      else [args.w]
    _rho_values    = args.rho_sweep    if args.rho_sweep    else [args.rho]
    _param_grid = [
        {"tau_ms": t, "w": w, "rho": r}
        for t, w, r in itertools.product(_tau_ms_values, _w_values, _rho_values)
    ]

    run_pipeline(
        cf=args.cf,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        output_dir=Path(args.output_dir)   if args.output_dir   else None,
        alpha=args.alpha,
        pref_dur=args.pref_dur,
        sigma_dur=args.sigma_dur,
        rectify=args.rectify,
        apply_adaptrans_flag=not args.no_adaptrans,
        param_grid=_param_grid,
        save_plots=not args.no_plots,
        noise_models=_noise_models,
    )
