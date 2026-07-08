# full_pipeline_shapekern_adaptrans.py
#
# Response-shape kernel variant of full_pipeline_with_adaptrans.py.
#
# By linearity, the AdapTrans ON response to a unit boxcar of duration D is
# the response-shape kernel h_ON_shape for any tone of that duration.
# This lets us replace the per-tone boxcar loop with a single convolution:
#
#   h_ON_shape  = AdapTrans_ON( unit_boxcar[0:D] )    # computed once
#   x[n]        = sum_s  prf_response[s] * delta[n - n_s^onset]
#   output[n]   = (x * h_ON_shape)[n]
#               = sum_s  prf_response[s] * h_ON_shape[n - n_s^onset]
#
# This is mathematically equivalent to the per-tone boxcar loop when all
# tones share the same duration (true for pure-tone DIPC sequences).
# For variable-duration sequences fall back to full_pipeline_with_adaptrans.py.

import math
import numpy as np
import sys
from pathlib import Path
import logging

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

logger = logging.getLogger(__name__)

EXP_NAME = "dipc_test_250225_01"
DEFAULT_BASE_DIR = Path(f"./models_output/{EXP_NAME}")
CHUNK_MARGIN_MS = 50.0    # extra window after tone offset used by chunk_from_id


def run_pipeline(
        exp_name: str = EXP_NAME,
        results_dir: Path = None,
        alpha: float = 2.0,
        pref_dur: float = 200.0,
        sigma_dur: float = 20.0,
        output_dir: Path = None,
        cf: int = 10,
        w: float = 0.8,
        K: int = None,
):
    _output_dir = output_dir or Path(f"./output/{exp_name}")
    LoggingConfigurator(
        output_dir=_output_dir,
        log_filename="prf_pipeline_shapekern_adaptrans.log",
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
        n_1ms = math.ceil(total_dur_ms)
        logger.debug("   total_dur_ms: %.1f ms | dt: %.4f ms", total_dur_ms, dt_s * 1000.0)

        # 3. Power-law sharpening
        sharpened_pop = apply_powerlaw_population(population_psth, alpha)
        sharpened = sharpened_pop[cf_index, :]

        # 4. Chunk into tone-on windows
        result, tone_dur_ms, isi_ms = chunk_from_id(sharpened, time_axis, seq_id, CHUNK_MARGIN_MS)
        n_tones = len(result["chunks"])
        logger.debug("  Tone dur: %.1f ms | ISI: %.1f ms | n_tones %d",
                     tone_dur_ms, isi_ms, n_tones)

        mean_rates_on = [np.mean(c) for c in result["chunks"]]

        # 5. Duration Gaussian (scalar)
        prf_responses = [
            apply_duration_gaussian_scalar(m, tone_dur_ms, pref_dur, sigma_dur)
            for m in mean_rates_on
        ]
        logger.debug("  pRF responses (first 3): %s", prf_responses[:3])

        # 6. Compute response-shape kernel from a unit boxcar of tone_dur_ms.
        #    The kernel captures the full temporal profile of AdapTrans ON
        #    for a sustained tone, so it can be used as a proper HRF-style kernel.
        unit_boxcar = np.zeros(n_1ms)
        i_off = round(tone_dur_ms)
        unit_boxcar[0:i_off] = 1.0

        on_off_unit = apply_adaptrans(
            unit_boxcar[np.newaxis, :],
            CFs_Hz=np.array([cf_hz]),
            dt_ms=1.0,
            w=w,
            K=K,
            pad_value=0.0,
        )
        h_ON_shape = on_off_unit[0, 0, :]   # shape (n_1ms,)
        logger.debug("  h_ON_shape max: %.4e | non-zero samples: %d",
                     h_ON_shape.max(), np.count_nonzero(h_ON_shape))

        # 7. Build scaled impulse train and convolve with h_ON_shape.
        #    x[n] = sum_s  prf_response[s] * delta[n - n_s^onset]
        #    output[n] = (x * h_ON_shape)[n]
        impulse_train = build_prf_impulse_train(
            prf_responses, result["onsets_ms"],
            total_dur_ms, dt_ms=1.0,
        )

        on_response = np.convolve(impulse_train, h_ON_shape, mode='full')[:n_1ms]

        logger.debug("  ON response shape: %s | min: %.4e | max: %.4e",
                     on_response.shape, on_response.min(), on_response.max())

    logger.info("Pipeline complete.")
    return {
        "prf_responses":  prf_responses,
        "impulse_train":  impulse_train,
        "h_ON_shape":     h_ON_shape,
        "on_response":    on_response,
        "tone_dur_ms":    tone_dur_ms,
        "chunk_margin_ms": CHUNK_MARGIN_MS,
    }


if __name__ == "__main__":
    run_pipeline()
