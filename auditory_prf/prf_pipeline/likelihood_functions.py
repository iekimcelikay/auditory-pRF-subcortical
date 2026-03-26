import numpy as np

# Created on 02/02/2026
# Ekim Celikay
# initial functions written by chatGPT.


# f(x;\theta)

def convolved_model_response ():
    return

def model_f(convolved_model_response, beta0, beta1, k, n):
    """
    """
    y_hat = beta0 + beta1 + convolved_model_response[k]**n
    return y_hat


# Log-likelihood under gaussian noise
# core statistical object
    # Chatgpt recommends
    # beta0, beta1, k, n, log_sigma = params
    # sigma = np.exp(log_sigma) and then return wouldbe
    # (-0.5 * N * np.log(2 * np.pi) - N * log_sigma -0.5 * np.sum(resid**2) / sigma**2)
    # enforce positivity of sigma

def log_likelihood(params, CF, y):
    """
    Gaussian log-likelihood with sigma estimated.
    params = [beta0, beta1, k, n, sigma]
    """
    beta0, beta1, k, n, sigma = params
    if sigma <= 0:
        return -np.inf

    y_hat = model_f(CF, beta0, beta1, k, n)
    resid = y - y_hat
    N = y.size

    return (
        -0.5 * N * np.log(2 * np.pi * sigma**2)
        -0.5 * np.sum(resid**2) / sigma**2
    )

def neg_log_likelihood(params, CF, y):
    return -log_likelihood(params, CF, y)

# Generative model for testing

def generate_data(CF, beta0, beta1, k, n, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    y_mean = model_f(CF, beta0, beta1, k, n)
    return y_mean + rng.normal(0.0, sigma, size=np.shape(y_mean))



# Fit parameters (maximum likelihood estimation)


## Now I do it myself
# Each voxel response, can be characterized with a 'k': cochlear channel response, and n: how much sharpening
# L = (beta0, beta1, k, n)
# If we find the 4 values for which L is minu=imum, you have the best of the models that explain the vlaues in this voxel.
# y = CF_k^n + Beta0 +