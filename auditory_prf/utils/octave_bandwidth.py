"""
Utility functions for octave bandwidth and Q factor calculations.
"""
import numpy as np


def octave_bandwidth(cf, x):
    """
    Calculate the frequency range for 1/X octave around a center frequency.
    
    Parameters
    ----------
    cf : float
        Center frequency in Hz
    x : float
        Octave divisor (e.g., x=3 for 1/3 octave bandwidth)
    
    Returns
    -------
    f_lower : float
        Lower frequency bound in Hz
    f_upper : float
        Upper frequency bound in Hz
    bandwidth : float
        Bandwidth in Hz (f_upper - f_lower)
    
    Examples
    --------
    >>> f_lower, f_upper, bw = octave_bandwidth(1000, 3)  # 1/3 octave at 1 kHz
    """
    f_lower = cf * 2**(-1/(2*x))
    f_upper = cf * 2**(1/(2*x))
    bandwidth = f_upper - f_lower
    return f_lower, f_upper, bandwidth


def octave_to_q(x):
    """
    Convert 1/X octave bandwidth to Q factor.
    
    Parameters
    ----------
    x : float
        Octave divisor (e.g., x=3 for 1/3 octave)
    
    Returns
    -------
    q : float
        Quality factor (Q)
    
    Examples
    --------
    >>> q = octave_to_q(3)  # Q factor for 1/3 octave
    """
    bw_oct = 1 / x
    q = np.sqrt(2**bw_oct) / (2**bw_oct - 1)
    return q


def q_to_octave(q):
    """
    Convert Q factor to octave bandwidth divisor.
    
    Parameters
    ----------
    q : float
        Quality factor
    
    Returns
    -------
    x : float
        Octave divisor for 1/X octave bandwidth
    
    Examples
    --------
    >>> x = q_to_octave(4.32)  # Get octave divisor from Q
    """
    bw_oct = 2 * np.log2(1/(2*q) + np.sqrt(1 + 1/(4*q**2)))
    x = 1 / bw_oct
    return x
