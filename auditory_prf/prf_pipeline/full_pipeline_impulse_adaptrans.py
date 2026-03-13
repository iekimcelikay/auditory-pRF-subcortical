# full_pipeline_impulse_adaptrans.py
#
# Impulse-train variant of full_pipeline_with_adaptrans.py.
#
# WARNING: This pipeline is kept for conceptual reference but does NOT produce
# the intended temporal dynamics of the AdapTrans ON filter.
#
# The AdapTrans h_ON kernel is a finite-difference / adaptation filter that
# compares the current input to an exponentially-weighted average of the recent
# past.  It requires a *sustained* (boxcar) input to express its onset-then-
# decay response.  A delta-spike input causes the ON channel to fire for exactly
# one sample and be zero everywhere else — no temporal evolution across the tone.
#
# The identity  output[n] = sum_s prf[s] * h_ON[n - n_s^onset]  only holds when
# h_ON is treated as a response-shape kernel (like an HRF), which AdapTrans is
# not.  Use full_pipeline_with_adaptrans.py (boxcar input) for correct results.
#
# Mathematical definition:
#   x[n]       = sum_s  prf_response[s] * delta[n - n_s^onset]
#   output[n]  = (x * h_ON)[n]
#              = sum_s  prf_response[s] * h_ON[n - n_s^onset]
#
# Difference from boxcar pipeline: each tone contributes a single delta spike
# at its onset, not a rectangular pulse over [onset, offset).

import math
import numpy as np
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.logging_configurator import LoggingConfigurator
from auditory_prf.utils.cochlea_loader_functions import (
    load_cochlea_results, organize_for_eachtone_allCFs, resolve_results_dir,
)
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import (
    load_cf_timecourse, load_population_psth,
)
from auditory_prf.prf_pipeline.powerlaw_function import (
    apply_power_normalize, apply_powerlaw_cf, apply_powerlaw_population,
)
from auditory_prf.prf_pipeline.chunk_timecourse import chunk_from_id
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import (
    build_prf_impulse_train, apply_adaptrans,
)

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
    _output_dir = output_dir or Path(f"./output/{exp_name}")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="prf_pipeline_impulse_adaptrans.log",
        file_level=logging.DEBUG,
        console_level=logging.DEBUG,
    ).setup()

    _results_dir = resolve_results_dir(
        results_dir if results_dir is not None else DEFAULT_BASE_DIR
    )
    logger.info("Experiment        : %s", exp_name)
    logger.info("Results directory : %s", _results_dir)

    npz_files = sorted(_results_dir.glob("*.npz"))
    if not npz_files:
        logger.error("No .npz files found in %s", _results_dir)
        sys.exit(1)
    logger.info("Found %d .npz file(s)", len(npz_files))

    for i, npz_path in enumerate(npz_files, 1):
        logger.info("[%d/%d] Processing: %s", i, len(npz_files), npz_path.name)

        population_psth, time_axis, cf_index, cf_hz, seq_id = load_population_psth(
            npz_path, cf
        )
        logger.debug("   CF index: %d | CF Hz: %.1f | seq_id: %s", cf_index, cf_hz, seq_id)
        dt_s = time_axis[1] - time_axis[0]
        total_dur_ms = (time_axis[-1] + dt_s) * 1000.0
        logger.debug("   total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

        sharpened_pop = apply_powerlaw_population(population_psth, alpha)
        sharpened = sharpened_pop[cf_index, :]
        logger.debug("  Sharpened timecourse shape: %s", sharpened.shape)

        result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id)
        n_tones = len(result["chunks"])
        logger.debug(
            "  Tone dur: %.1f ms | ISI: %.1f ms | n_tones %d",
            tone_dur_ms, isi_ms, n_tones,
        )

        mean_rates_on = [np.mean(c) for c in result["chunks"]]

        prf_responses = [
            apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur, sigma_dur)
            for m in mean_rates_on
        ]
        logger.debug("  pRF responses (first 3): %s", prf_responses[:3])

        # Per-tone AdapTrans ON filter using impulse train (delta at onset only).
        # x[n] = sum_s  prf_response[s] * delta[n - n_s^onset]
        n_1ms = math.ceil(total_dur_ms)
        on_response = np.zeros(n_1ms)
        train       = np.zeros(n_1ms)  # accumulated impulse train for diagnostics

        for s, (prf_s, on_ms) in enumerate(
                zip(prf_responses, result["onsets_ms"])):

            # single delta spike at tone onset
            single_train = build_prf_impulse_train(
                [prf_s], np.array([on_ms]),
                total_dur_ms, dt_ms=1.0,
            )
            train += single_train

            # apply AdapTrans ON; pad_value=0.0 → silence before t=0
            on_off_s = apply_adaptrans(
                single_train[np.newaxis, :],
                CFs_Hz=np.array([cf_hz]),
                dt_ms=1.0,
                w=w,
                K=K,
                pad_value=0.0,
            )
            on_response += on_off_s[0, 0, :]

        logger.debug(
            "  ON response (summed %d tones) shape: %s | min: %.4e | max: %.4e",
            n_tones, on_response.shape, on_response.min(), on_response.max(),
        )

    logger.info("Pipeline complete.")
    return {"prf_responses": prf_responses, "on_response": on_response, "train": train}


if __name__ == "__main__":
    run_pipeline()