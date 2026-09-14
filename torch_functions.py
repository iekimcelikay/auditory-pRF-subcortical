import torch


def soft_select_channel_linear(population_psth_tensor, cf_hz_continuous, cf_list):
    """Linearly interpolate a continuous CF between its two neighboring
    cochlear channels, in log-Hz space.

    Parameters
    ----------
    population_psth_tensor : (n_channels, T) tensor
        Raw (unsharpened) population response for one stimulus sequence.
    cf_hz_continuous : scalar tensor
        This voxel's fitted CF, in Hz (already passed through its bounded
        reparam, so it's guaranteed to lie within [cf_list.min(), cf_list.max()]).
    cf_list : (n_channels,) tensor
        Each channel's CF in Hz, sorted ascending.

    Returns
    -------
    (T,) tensor — collapsed raw timecourse for this voxel's continuous CF.
    """
    log_freqs = cf_list.log()
    log_cf    = cf_hz_continuous.log()

    # searchsorted just picks which two channels bracket cf_hz_continuous —
    # that choice isn't itself differentiable, only `frac` needs to be.
    i1 = torch.searchsorted(log_freqs, log_cf.detach()).clamp(1, len(log_freqs) - 1)
    i0 = i1 - 1

    f0, f1 = log_freqs[i0], log_freqs[i1]
    frac = ((log_cf - f0) / (f1 - f0)).clamp(0.0, 1.0)

    return (1 - frac) * population_psth_tensor[i0] + frac * population_psth_tensor[i1]


def apply_powerlaw_torch(x, alpha):
    pre_mean  = x.mean()
    sharpened = x.pow(alpha)
    post_mean = sharpened.mean()
    return sharpened * (pre_mean / (post_mean + 1e-10))