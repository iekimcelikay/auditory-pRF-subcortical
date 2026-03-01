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

# Package level imports
from auditory_prf.utils.result_saver import ResultSaver
from auditory_prf.utils.cochlea_loader_functions import load_cochlea_results, organize_for_eachtone_allCFs, resolve_results_dir
from auditory_prf.prf_pipeline.load_extract_cf_timecourse import load_cf_timecourse, get_cf_timecourse
from auditory_prf.prf_pipeline.powerlaw_function import apply_power_normalize, apply_powerlaw_cf
# Duration (scalar)
from auditory_prf.prf_pipeline.duration_models import apply_duration_gaussian_scalar

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
# ---- 4. Tone-ON chunk timecourse (TODO:TO BE WRITTEN)
# script: chunk_timecourse.py TODO: TO BE WRITTEN
# PLAN:
# Add a chunk_timecourse function.
# It takes timecourse (n_bins,), time_axis (n_bins,), tone_onset_s and tone_offset_s,
# and returns (mean_rate_on, tone_dur).
# `tone_dur = tone_offset_s - tone_onset_s`
# Select bins where tone_onset_s <= time_axis < tone_offset_s
# Return np.mean(timecourse[mask]) and tone_dur.
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
#===== 2. EXTRACT ONE TIME COURSE
get_cf_timecourse()
#===== 3. APPLY SHARPENING (LATERAL INHIBITION) WITH ALPHA
apply_power_normalize()

# sharpened = apply_powerlaw_cf(timecourse, alpha)
#
#
#===== 4. CUT TO CHUNKS FOR TONE-ON and TONE-OFF, TAKE THE AVERAGE FIRING RATE = 1 VALUE ACTING AS  A WEIGHT
# Chunk the power-normalized 1-D timecourse into tone-on / tone-off windows;
# compute the mean firint rate for the each tone-on windows -> scalar mean_rate_on; and record the window's duration -> scalar tone_dur
# mean_rate_on, tone_dur = chunk_timecourse(sharpened, time_axis, tone_onset_s, tone_offset_s)
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
apply_duration_gaussian_scalar()
# prf_response = apply_duration_gaussian_scalar(mean_rate_on, tone_dur, pref_dur, sigma_dur)



