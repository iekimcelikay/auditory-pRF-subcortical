import numpy as np

def gaussian_duration(stim_dur: float, pref_dur: float, sigma_dur: float) -> float:
    """
    Gaussian over stimulus duration d.
    LaTeX: f(d) = (1 / sqrt(2*pi)*sigma_d) * exp(-(d - d_hat)^2 / 2*sigma_d^2)

    stim_dur  : duration of the stimulus in seconds
    pref_dur  : preferred duration d_hat (tau0) — parameter to fit
    sigma_dur : duration tuning width sigma_d (sigma_tau0) — parameter to fit
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma_dur)) * \
        np.exp(-(stim_dur - pref_dur)**2 / (2 * sigma_dur**2))

# OPTION A: SCALAR WEIGHT
# One gaussian value per stimulus, multiplied against the
# mean firing rate scalar from step 4 (full_pipeline_with_adaptrans.py)
def apply_duration_gaussian_scalar(mean_rate_on: float, stim_dur: float,
                                    pref_dur: float, sigma_dur: float) -> float:
    """
    Computes a single Gaussian weight for the stimulus duration,
    then scales the mean firing rate by it.

    mean_rate_on : scalar average firing rate during tone-on window (from step 4)
    stim_dur     : duration of the stimulus in seconds (e.g. tone_offset - tone_onset)
    pref_dur     : preferred duration (tau0) — parameter to fit
    sigma_dur    : duration tuning width (sigma_tau0) — parameter to fit

    Returns a single scalar.
    -------
    prf_response : float
        mean_rate_on * gaussian_duration(tone_dur, pref_dur, sigma_dur)
    """
    weight = gaussian_duration(stim_dur, pref_dur, sigma_dur)
    return mean_rate_on * weight

# OPTION B: POINTWISE MULTIPLICATION
# Gaussian evaluated at each time bin, multiplied against the
# full power-normalized time course from step 3 (full_pipeline_without_adaptrans.py)
def apply_duration_gaussian_pointwise(timecourse: np.ndarray, time_axis: np.ndarray,
                                       pref_dur: float, sigma_dur: float) -> np.ndarray:
    """
    Evaluates the Gaussian at every time point (treating each time bin's
    position as a 'duration'), then multiplies pointwise against the timecourse.

    timecourse : shape (n_bins,) — power-normalized output from step 3
    time_axis  : shape (n_bins,) — time in seconds for each bin
    pref_dur   : preferred duration (tau0) — parameter to fit
    sigma_dur  : duration tuning width (sigma_tau0) — parameter to fit

    Returns array of shape (n_bins,).
    """
    weights = gaussian_duration(time_axis, pref_dur, sigma_dur)  # shape (n_bins,)
    return timecourse * weights