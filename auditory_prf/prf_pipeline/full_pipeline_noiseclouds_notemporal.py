"""full_pipeline_noiseclouds_notemporal.py
=========================================
pRF pipeline for noise-cloud (Gaussian filterbank) stimuli, without temporal models.

Noise-cloud stimuli use a different naming convention than tone-cloud stimuli:
    noisecloud{g:02d}_fc{center_hz}hz_bw{bw:.2f}oct_dur{D}ms_isi{I}ms_numtones{N}

This convention is stored verbatim as `soundfileid` in each cochlear NPZ file, so the
seq_id keys used here match directly — no `make_condition_map` lookup table is needed.

Skipped stages (vs full_pipeline_with_adaptrans):
  - Duration Gaussian filter  (no pref_dur / sigma_dur)
  - AdapTrans ON/OFF filter    (no onset/offset decomposition)

Retained stages:
  1. Load cochlea PSTH results
  2. Power-law sharpening (alpha)
  3. Chunk into tone-ON windows -> mean rate per tone
  4. Build boxcar train (amplitude = mean rate)
  5. Assemble N runs from run design
  6. HRF convolution -> BOLD
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np

from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_population_psth
from auditory_prf.prf_pipeline.powerlaw_function import apply_powerlaw_population
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.run_assembly import (
    generate_run_design, assemble_run_bold, apply_run_noise, parse_noise_seed_arg,
)
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from prf_models.pm_noise import PmNoise

logger = logging.getLogger(__name__)

# ── multiprocessing worker (module-level so it is picklable) ─────────────────
_worker: dict = {}


def _worker_init(per_seq, hrf_kernel, total_run_dur_s, cf_hz, tr_s, signal_dt_s, noise_model):
    _worker["per_seq"]         = per_seq
    _worker["hrf_kernel"]      = hrf_kernel
    _worker["total_run_dur_s"] = total_run_dur_s
    _worker["cf_hz"]           = cf_hz
    _worker["tr_s"]            = tr_s
    _worker["signal_dt_s"]     = signal_dt_s
    _worker["noise_model"]     = noise_model


def _assemble_one(task: tuple) -> tuple:
    run_idx, run_design = task
    result = assemble_run_bold(
        per_seq          = _worker["per_seq"],
        run_design       = run_design,
        total_run_dur_s  = _worker["total_run_dur_s"],
        hrf_kernel       = _worker["hrf_kernel"],
        cf_hz            = _worker["cf_hz"],
        tr_s             = _worker["tr_s"],
        signal_dt_s      = _worker["signal_dt_s"],
        apply_adaptrans_flag=False,
    )
    bold_noisy = apply_run_noise(
        result["bold_combined"], _worker["noise_model"], run_idx, _worker["tr_s"]
    )
    return run_idx, run_design, result["bold_combined"], bold_noisy, result["bold_on"], result["t_tr"]


# ── noise-cloud seq_id scheme ─────────────────────────────────────────────────
BAND_CENTERS_HZ   = (572, 885, 1322)   # ERB-spaced Gaussian filterbank centers (450-1600 Hz, N=3)
BW_OCT            = 0.50               # burst bandwidth in octaves
TOTAL_SEQ_DUR_S   = 20.0               # WAV sequence duration (seconds)
STIMULUS_SAMPLE_RATE = 100_000         # sample rate used during WAV generation
NC_SILENCE_SEQ_ID = "noisecloud00_dur0ms_isi0ms"


def _make_noisecloud_seq_id_fn(
    band_centers_hz: tuple,
    bw_oct: float,
    total_seq_dur_s: float,
    sample_rate: int,
):
    """Return a callable ``(dur_ms, isi_ms, g_idx) -> seq_id str``.

    Reproduces the filename produced by save_noise_clouds_gaussian_prf.py:
        noisecloud{g:02d}_fc{hz}hz_bw{bw:.2f}oct_dur{D}ms_isi{I}ms_numtones{N}

    numtones is computed with the same sample-accurate floor-division formula as
    calculate_num_tones() in stimuli/sample_tone_cloud_freqs_gaussian.py so that
    keys match the stored soundfileid exactly. Pass float dur_ms values (not
    rounded integers) to avoid off-by-one rounding errors for long durations.

    ``g_idx=None`` returns NC_SILENCE_SEQ_ID (null / silence trial).
    """
    total_samples = int(total_seq_dur_s * sample_rate)

    def _fn(dur_ms, isi_ms, g_idx):
        if g_idx is None:
            return NC_SILENCE_SEQ_ID
        center_hz   = band_centers_hz[g_idx]
        tone_samples = int(dur_ms / 1000.0 * sample_rate)
        isi_samples  = int(isi_ms / 1000.0 * sample_rate)
        numtones     = total_samples // (tone_samples + isi_samples)
        return (
            f"noisecloud{g_idx + 1:02d}"
            f"_fc{int(round(center_hz))}hz"
            f"_bw{bw_oct:.2f}oct"
            f"_dur{int(round(dur_ms))}ms"
            f"_isi{int(round(isi_ms))}ms"
            f"_numtones{numtones}"
        )
    return _fn


# ── experiment defaults ───────────────────────────────────────────────────────
EXP_NAME         = "noiseclouds_gaussianprf"
DEFAULT_BASE_DIR  = Path(f"./models_output/{EXP_NAME}")

# ── run design defaults ───────────────────────────────────────────────────────
# Float durations from find_closest_durations() in find_optimal_durations.py.
# Must be floats (not rounded ints) so that numtones computation matches
# calculate_num_tones() used during WAV generation.
TONE_ON_MS       = (35.14, 44.93, 60.0, 75.44, 100.0, 150.0, 250.88, 488.24)
ISI_MS           = (100,) * len(TONE_ON_MS)
NULL_FRACTION    = 0.25
TRIAL_DURATION_S  = 20.0
OPENING_BLANK_S   = 10.0
CLOSING_BLANK_S   = 10.0
ITI_RANGE_S       = 0
N_RUNS            = 4
BASE_SEED         = 42


def run_pipeline(
        exp_name: str = EXP_NAME,
        results_dir: Optional[Path] = None,
        alpha: float = 2.0,
        cf=10,
        output_dir: Optional[Path] = None,
        # noise-cloud filterbank
        band_centers_hz: tuple = BAND_CENTERS_HZ,
        bw_oct: float = BW_OCT,
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
        # parallelism
        n_workers: int = 1,
        # noise
        noise_model: Optional[PmNoise] = None,
):
    _output_dir = output_dir or Path(f"./output/{exp_name}_noiseclouds_notemporal")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="pipeline_noiseclouds_notemporal.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    ).setup()

    _results_dir = (results_dir if results_dir is not None else DEFAULT_BASE_DIR)
    _results_dir = Path(_results_dir).expanduser().resolve()
    logger.info("Experiment       : %s", exp_name)
    logger.info("Results dir      : %s", _results_dir)
    logger.info("Band centres (Hz): %s", band_centers_hz)
    logger.info("BW (oct)         : %.2f", bw_oct)

    npz_files = sorted(_results_dir.glob("wav*/**/*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {_results_dir}/wav*/. "
            "Run the cochlear simulation first and check results_dir."
        )
    logger.info("Found %d NPZ file(s)", len(npz_files))

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

    seq_id_fn = _make_noisecloud_seq_id_fn(
        band_centers_hz, bw_oct, total_seq_dur_s, stimulus_sample_rate
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

        cf_tc_raw = population_psth[cf_index, :]

        if seq_id == NC_SILENCE_SEQ_ID:
            spont_rate = float(np.mean(cf_tc_raw))
            n_samples  = int(round(total_dur_ms))
            train      = np.full(n_samples, spont_rate)
            logger.debug("  seq_id=%s | CF=%.0f Hz | spont_rate=%.2f sp/s | train len=%d",
                         seq_id, cf_hz, spont_rate, len(train))
        else:
            result, _, _ = chunk_from_id(cf_tc_raw, time_axis, seq_id)
            mean_rates_on = apply_powerlaw_population(
                np.array([np.mean(c) for c in result["chunks"]]), alpha
            )

            train = build_prf_boxcar_train(
                mean_rates_on,
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
                        for ton, isi, g_idx in stimuli} | {NC_SILENCE_SEQ_ID}
    missing = expected_seq_ids - set(per_seq.keys())
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"Stimulus params do not match .npz files in results_dir.\n"
            f"{len(missing)} expected seq_ids not found in per_seq "
            f"(e.g. {sample}).\n"
            f"Check that BAND_CENTERS_HZ, BW_OCT, TONE_ON_MS, and ISI_MS match "
            f"the WAV files used to generate the cochlear results."
        )

    # ── Phase 2: per-run assembly ──────────────────────────────────────────────
    cf_hz_used = next(iter(per_seq.values()))["cf_hz"]
    all_runs: dict = {}

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
    logger.info("Phase 2: assembling %d run(s) with %d worker(s).",
                len(_designs), n_workers)

    tasks = list(enumerate(_designs))

    if n_workers > 1:
        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(per_seq, hrf_kernel, total_run_dur_s,
                      cf_hz_used, tr_s, signal_dt_s, noise_model),
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
                apply_adaptrans_flag=False,
            )
            bold_noisy = apply_run_noise(result["bold_combined"], noise_model, run_idx, tr_s)
            results.append((run_idx, run_design, result["bold_combined"], bold_noisy,
                             result["bold_on"], result["t_tr"]))

    bold_combined = None
    for run_idx, run_design, bold_combined, bold_noisy, bold_on, t_tr in results:
        all_runs[f"run_{run_idx + 1:02d}"] = {
            "run_design":    run_design,
            "bold_combined": bold_combined,
            "bold_noisy":    bold_noisy,
            "bold_on":       bold_on,
            "t_tr":          t_tr,
            "seed":          base_seed + run_idx if run_designs is None else None,
        }
    if bold_combined is not None:
        logger.info("  BOLD shape: %s", bold_combined.shape)

    # ── Save ──────────────────────────────────────────────────────────────────
    saver = ResultSaver(_output_dir)
    save_dict = {
        "exp_name": exp_name,
        "cf":       cf,
        "alpha":    alpha,
        "tr_s":     tr_s,
        **{k: v["bold_combined"] for k, v in all_runs.items()},
    }
    if noise_model is not None:
        save_dict.update({f"{k}_noisy": v["bold_noisy"] for k, v in all_runs.items()})
        save_dict["noise_seed"] = str(noise_model.seed)
    saver.save_npz(
        save_dict,
        f"{exp_name}_noiseclouds_notemporal_cf{cf:03d}_bold.npz",
    )
    logger.info("Saved BOLD to %s", _output_dir)

    return all_runs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=int, default=0,
                        help="CF index (0-based). Pass $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to cochlea NPZ results directory.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--noise_voxel", choices=["none", "low", "mid", "high"],
                        default="none",
                        help="BOLD noise preset (PmNoise voxel level). "
                             "'none' = no noise (default).")
    parser.add_argument("--noise_seed", type=str, default="random",
                        help="PmNoise seed: an integer for reproducible noise, "
                             "'random' (default), or 'none'/'nonoise'.")
    args = parser.parse_args()

    _noise_model = None
    if args.noise_voxel != "none":
        _noise_model = PmNoise(voxel=args.noise_voxel, seed=parse_noise_seed_arg(args.noise_seed))

    run_pipeline(
        cf=args.cf,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        output_dir=Path(args.output_dir)   if args.output_dir   else None,
        alpha=args.alpha,
        noise_model=_noise_model,
    )
