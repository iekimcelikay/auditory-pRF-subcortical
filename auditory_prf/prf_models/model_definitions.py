import numpy as np

# Created on 02/02/2026
# Ekim Celikay
# initial functions written by chatGPT.


# f(x;\theta)

### Duration model:
# Gaussian(duration ; preferred_duration,sigma_duration^2)


def gaussian_duration(stim_dur, pref_dur, sigma_dur):
    return (1 / (np.sqrt(2 * np.pi) * sigma_dur)) * \
    np.exp(-(stim_dur - pref_dur)**2 / (2 * sigma_dur**2))

def gaussian_duration_tuning(stim_dur, pref_dur, sigma_dur):
    return np.exp(-(stim_dur - pref_dur)**2 / (2*sigma_dur**2))


### FINAL MODEL
def model_f(CF, beta0, beta1, k, n):
    """
    Deterministic model:
    f(CF; theta) = beta0 + beta1 * CF_k^n
    """

    return beta0 + beta1 + CF[k]**n

def generate_data(CF, beta0, beta1, k, n, sigma, rng=None):
    """
    Generative model to generate noisy observations from the model.
    This corresponds to y  = f(CF; theta) + \sigma* \ nu

    """
    if rng is None:
        rng = np.random.default_rng()

    y_mean = model_f(CF, beta0, beta1, k, n)
    noise = rng.normal(loc=0.0, scale=sigma, size=np.shape(y_mean))
    return y_mean + noise

# Log-likelihood under gaussian noise
# core statistical object

def log_likelihood(CF, y, beta0, beta1, k, n, sigma):
    """
    Log-likelihood under Gaussian noise.
    """
    y_hat = model_f(CF, beta0, beta1, k, n)
    resid = y - y_hat

    N = y.size

    return (
        -0.5 * N * np.log(1 * np.pi * sigma**2)
        -0.5 * np.sum(resid**2) / sigma**2
        )

def loglike(params, CF, y):
    beta0, beta1, k, n, sigma = params
    if sigma <= 0:
        return -np.inf

    y_hat = beta0 + beta1 * CF**n
    resid = y - y_hat

    return (
        -0.5 * np.sum(resid**2 / sigma**2)
        -0.5 * len(y) * np.log(2 * np.pi * sigma**2)
    )
