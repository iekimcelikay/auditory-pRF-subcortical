# full_pipeline_with_adaptrans.py
#
# PARAMETERS TO FIT = {cf_index, stimulus_id, sharpening_factor, preferred_duration, sigma_duration}
# Parameter to fit = Theta
# cf_index = k
# stimulus_id = s,
# sharpening_factor = alpha,
# preferred_duration = tau0,
# sigma_duration = sigma_tau0


import math
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

# AdapTrans ON filter
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train, apply_adaptrans

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
):

    # --- Logging setup -----
    _output_dir = output_dir or Path(f"./output/{exp_name}")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="prf_pipeline_with_adaptrans.log",
        file_level=logging.DEBUG,
        console_level=logging.DEBUG,
        ).setup()

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

        # DEBUG STEP _ DELETE LATER
        plt.figure(figsize=(12, 4))
        plt.plot(sharpened, linewidth=0.8)
        plt.title('Sharpened Array')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.show()

        # 4. Chunk into tone-on windows
        result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id)
        n_tones = len(result["chunks"])
        logger.debug("  Tone dur: %.1f ms | ISI: %.1f ms | n_tones %d",
                     tone_dur_ms, isi_ms, n_tones)


        # get the mean rates of tone-on chunks
        mean_rates_on = [np.mean(c) for c in result["chunks"]]
        # TODO: Add a loger.debug here that prints mean_rates_on for each tone.
        print([m for m in mean_rates_on])

        # 5. Apply duration Gaussian (Scalar)
        prf_responses = [
            apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur, sigma_dur)
            for m in mean_rates_on
            ]
        logger.debug("  pRF responses(first 3): %s", prf_responses[:3])

        # 6-7. Per-tone AdapTrans ON filter, then sum
        #   For each tone s:
        #     - build an isolated boxcar: zeros everywhere except [onset_s, offset_s)
        #     - apply AdapTrans ON to that single-tone signal
        #     - accumulate into on_response (superposition)
        #   train is also accumulated (summed boxcar) for diagnostics/plotting.
        n_1ms = math.ceil(total_dur_ms)
        on_response = np.zeros(n_1ms)
        train       = np.zeros(n_1ms)

        for s, (prf_s, on_ms, off_ms) in enumerate(
                zip(prf_responses, result["onsets_ms"], result["offsets_ms"])):

            # isolated single-tone boxcar
            single_train = build_prf_boxcar_train(
                [prf_s], np.array([on_ms]), np.array([off_ms]),
                total_dur_ms, dt_ms=1.0,
            )
            train += single_train   # accumulate for visualisation

            # AdapTrans ON filter on this tone in isolation.
            # pad_value=0.0 ensures silence is assumed before t=0,
            # even for tone 1 which starts at t=0 (signal[0] = amp there).
            on_off_s = apply_adaptrans(
                single_train[np.newaxis, :],
                CFs_Hz=np.array([cf_hz]),
                dt_ms=1.0,
                w=w,
                K=K,
                pad_value=0.0,
            )
            on_response += on_off_s[0, 0, :]   # superpose

        logger.debug("  ON response (summed %d tones) shape: %s | min: %.4e | max: %.4e",
                     n_tones, on_response.shape, on_response.min(), on_response.max())

    logger.info("Pipeline complete.")
    return {"prf_responses": prf_responses, "on_response": on_response, "train": train}


if __name__ == "__main__":
    run_pipeline()
