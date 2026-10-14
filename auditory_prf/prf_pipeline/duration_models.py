import numpy as np

def gaussian_duration(stim_dur: float, pref_dur: float, sigma_dur: float) -> float:
    """
    Gaussian TUNING CURVE over stimulus duration d, peak height 1.0 at d ==
    pref_dur (per project discussion: the previous version used a normalized
    Gaussian PROBABILITY DENSITY formula, 1/(sqrt(2*pi)*sigma_d), whose peak
    height is sigma_d-dependent and far below 1 for realistic sigma_d -- a
    stimulus at the EXACT preferred duration still got crushed by a factor
    of ~1/(sigma_d*sqrt(2*pi)) instead of passing through unweighted. That's
    correct for a probability density (which must integrate to 1), but wrong
    for a response weight/gain, which should leave a perfectly-matched
    stimulus at full strength.
    LaTeX: f(d) = exp(-(d - d_hat)^2 / 2*sigma_d^2)

    stim_dur  : duration of the stimulus in seconds
    pref_dur  : preferred duration d_hat (tau0) — parameter to fit
    sigma_dur : duration tuning width sigma_d (sigma_tau0) — parameter to fit
    """
    return np.exp(-(stim_dur - pref_dur)**2 / (2 * sigma_dur**2))

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

