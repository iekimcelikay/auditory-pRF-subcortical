# ORIGINAL CODE FROM deepSTRF
# TODO 
# [ ] add proper citations
# https://github.com/urancon/deepSTRF/blob/develop/deepSTRF/models/prefiltering.py#L229
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def tau_to_a(time_constants, dt: float = 0.001):
    """Convert physical time constants (s) to dimensionless ``a`` parameters. 

    Parameters
    ----------
        time_constants (torch.Tensor): 
            Time constants in seconds.
        dt : float, default=0.001 
            Time step width in seconds.

    Returns
    --------
        torch.Tensor: 
            The corresponding ``a = exp(-dt / tau)`` parameters.
    """
    return torch.exp(-dt / time_constants)

def a_to_tau(a, dt: float = 0.001):

    return -dt / torch.log(a)

