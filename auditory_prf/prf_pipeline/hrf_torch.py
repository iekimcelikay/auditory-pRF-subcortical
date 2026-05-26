"""hrf_torch.py
==============
Differentiable double-gamma HRF for PyTorch optimization.

Learnable per-voxel HRF parameters using nn.Parameter and torch.lgamma for
numerical stability. Designed to integrate with AuditoryVoxelModel for
gradient-based optimization of HRF kernels.

Classes:
    HRFKernel           — nn.Module with 5 learnable HRF params per voxel

Helper functions:
    build_hrf_kernel_torch(...)     -> (kernel, t)
    convolve_hrf_torch_causal(...)  -> convolved signal

Presets:
    SPM_PARAMS, GLOVER_PARAMS, POPEYE_PARAMS, SUBCORTICAL_PARAMS
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset parameter dicts
# ---------------------------------------------------------------------------

SPM_PARAMS = dict(
    peak_delay=6.0,
    peak_disp=1.0,
    under_delay=16.0,
    under_disp=1.0,
    p_u_ratio=6.0,
)

GLOVER_PARAMS = dict(
    peak_delay=5.4,
    peak_disp=0.9,
    under_delay=10.8,
    under_disp=0.9,
    p_u_ratio=1.0 / 0.35,
)

POPEYE_PARAMS = dict(
    peak_delay=5.4,
    peak_disp=0.9,
    under_delay=10.9,
    under_disp=0.9,
    p_u_ratio=6.0,
)

SUBCORTICAL_PARAMS = dict(
    peak_delay=5.0,
    peak_disp=1.0,
    under_delay=9.0,
    under_disp=1.0,
    p_u_ratio=6.0,
)


# ---------------------------------------------------------------------------
# Stable gamma PDF using torch.lgamma
# ---------------------------------------------------------------------------

def gamma_pdf_torch(
    x: torch.Tensor,
    shape: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Compute gamma PDF using log-gamma for numerical stability.

    PDF(x; a, s) = (1 / (s^a * Gamma(a))) * x^(a-1) * exp(-x/s)

    Using logs:
    log PDF = (a-1)*log(x) - x/s - a*log(s) - log(Gamma(a))

    Parameters
    ----------
    x : torch.Tensor, shape (n,)
        Evaluation points (must be positive).
    shape : torch.Tensor, scalar or broadcastable
        Shape parameter a (> 0).
    scale : torch.Tensor, scalar or broadcastable
        Scale parameter s (> 0).

    Returns
    -------
    pdf : torch.Tensor, shape (n,)
        Gamma PDF values (non-negative).
    """
    # Avoid log(0) and log(negative)
    x = torch.clamp(x, min=1e-8)

    # log PDF = (a-1)*log(x) - x/s - a*log(s) - log(Gamma(a))
    log_pdf = (shape - 1.0) * torch.log(x) - x / scale - shape * torch.log(scale) - torch.lgamma(shape)

    return torch.exp(log_pdf)


# ---------------------------------------------------------------------------
# Kernel building
# ---------------------------------------------------------------------------

def build_hrf_kernel_torch(
    peak_delay: torch.Tensor | float = 6.0,
    peak_disp: torch.Tensor | float = 1.0,
    under_delay: torch.Tensor | float = 16.0,
    under_disp: torch.Tensor | float = 1.0,
    p_u_ratio: torch.Tensor | float = 6.0,
    dt: float = 0.001,
    duration: float = 32.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a normalised double-gamma HRF kernel in PyTorch.

    h(t) = gamma(peak_delay, peak_disp)(t)
         - gamma(under_delay, under_disp)(t) / p_u_ratio

    Normalised so the positive lobe has area = 1.

    Parameters
    ----------
    peak_delay : float or torch.Tensor
        Delay of the peak gamma (seconds). Default: 6.
    peak_disp : float or torch.Tensor
        Dispersion (scale) of the peak gamma. Default: 1.
    under_delay : float or torch.Tensor
        Delay of the undershoot gamma (seconds). Default: 16.
    under_disp : float or torch.Tensor
        Dispersion of the undershoot gamma. Default: 1.
    p_u_ratio : float or torch.Tensor
        Peak-to-undershoot ratio. Larger = smaller undershoot. Default: 6.
    dt : float
        Time resolution in seconds (default 0.001 = 1 ms).
    duration : float
        Kernel duration in seconds (default 32).
    device : torch.device, optional
        Device for tensor creation. None uses default.

    Returns
    -------
    kernel : torch.Tensor, shape (n_samples,)
        Normalised HRF kernel.
    t : torch.Tensor, shape (n_samples,)
        Time axis in seconds.
    """
    if device is None:
        device = torch.device("cpu")

    # Convert scalars to tensors if needed
    if not isinstance(peak_delay, torch.Tensor):
        peak_delay = torch.tensor(peak_delay, device=device, dtype=torch.float32)
    if not isinstance(peak_disp, torch.Tensor):
        peak_disp = torch.tensor(peak_disp, device=device, dtype=torch.float32)
    if not isinstance(under_delay, torch.Tensor):
        under_delay = torch.tensor(under_delay, device=device, dtype=torch.float32)
    if not isinstance(under_disp, torch.Tensor):
        under_disp = torch.tensor(under_disp, device=device, dtype=torch.float32)
    if not isinstance(p_u_ratio, torch.Tensor):
        p_u_ratio = torch.tensor(p_u_ratio, device=device, dtype=torch.float32)

    # Time axis
    t = torch.arange(0.0, duration, dt, device=device, dtype=torch.float32)

    # Gamma shape and scale
    peak_shape = peak_delay / peak_disp
    under_shape = under_delay / under_disp

    # Compute gamma PDFs
    peak = gamma_pdf_torch(t, peak_shape, peak_disp)
    under = gamma_pdf_torch(t, under_shape, under_disp)

    # Double-gamma kernel
    kernel = peak - under / p_u_ratio

    # Normalise by positive lobe area
    pos_mask = kernel > 0
    if pos_mask.any():
        pos_area = torch.sum(kernel[pos_mask]) * dt
        if pos_area > 0:
            kernel = kernel / pos_area

    return kernel, t


# ---------------------------------------------------------------------------
# Convolution (causal)
# ---------------------------------------------------------------------------

def convolve_hrf_torch_causal(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    signal_dt: float = 0.001,
) -> torch.Tensor:
    """Convolve a neural impulse train with HRF kernel (causal, preserves length).

    Left-pads the signal by (kernel_len - 1), convolves, then trims back to
    original length. This ensures causality (kernel only looks at past) and
    preserves the output length.

    Parameters
    ----------
    signal : torch.Tensor, shape (n,) or (batch, n)
        Neural signal (1-D or batched).
    kernel : torch.Tensor, shape (k,)
        HRF kernel (from build_hrf_kernel_torch).
    signal_dt : float
        Time resolution of signal in seconds (default 0.001 = 1 ms).
        Used to scale the convolution by dt for numerical correctness.

    Returns
    -------
    conv : torch.Tensor, shape same as input
        Convolved signal, same length as input.
    """
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)  # (1, n)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, n_samples = signal.shape
    kernel_len = kernel.shape[0]

    # Left-pad signal with zeros (causal: kernel looks backward)
    padded_signal = torch.nn.functional.pad(signal, (kernel_len - 1, 0), mode="constant", value=0.0)
    # padded_signal: (batch, n + kernel_len - 1)

    # Reshape for grouped convolution (treat batch as separate channels)
    padded_signal = padded_signal.unsqueeze(1)  # (batch, 1, n + kernel_len - 1)
    kernel_1d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_len)

    # Convolve
    conv = torch.nn.functional.conv1d(padded_signal, kernel_1d, padding=0)
    # conv: (batch, 1, n)

    # Scale by dt
    conv = conv * signal_dt

    # Remove batch dimension if input was 1-D
    conv = conv.squeeze(1)  # (batch, n)
    if squeeze_output:
        conv = conv.squeeze(0)  # (n,)

    return conv


# ---------------------------------------------------------------------------
# HRFKernel: Learnable nn.Module for per-voxel HRF parameters
# ---------------------------------------------------------------------------

class HRFKernel(nn.Module):
    """Learnable double-gamma HRF kernel with per-voxel parameters.

    Each voxel has 5 learnable HRF parameters: peak_delay, peak_disp,
    under_delay, under_disp, p_u_ratio. The forward pass builds and returns
    the normalised kernel for each voxel.

    Attributes
    ----------
    n_voxels : int
        Number of voxels.
    peak_delay : nn.Parameter, shape (n_voxels,)
        Learnable peak delay (seconds).
    peak_disp : nn.Parameter, shape (n_voxels,)
        Learnable peak dispersion.
    under_delay : nn.Parameter, shape (n_voxels,)
        Learnable undershoot delay (seconds).
    under_disp : nn.Parameter, shape (n_voxels,)
        Learnable undershoot dispersion.
    p_u_ratio : nn.Parameter, shape (n_voxels,)
        Learnable peak-to-undershoot ratio.

    Methods
    -------
    forward(signal, signal_dt=0.001)
        Convolve signal with per-voxel HRF kernels.
    get_kernels(dt, duration)
        Return all per-voxel HRF kernels (for inspection/debugging).
    """

    def __init__(
        self,
        n_voxels: int,
        init_preset: str = "subcortical",
        dt: float = 0.001,
        duration: float = 32.0,
        device: Optional[torch.device] = None,
    ):
        """Initialise HRFKernel module.

        Parameters
        ----------
        n_voxels : int
            Number of voxels (learnable parameter sets).
        init_preset : str
            Preset to use for initialisation: 'spm', 'glover', 'popeye', 'subcortical'.
            Default: 'subcortical' (appropriate for cochlear nucleus / inferior colliculus).
        dt : float
            Time resolution for kernel building (seconds). Default: 0.001.
        duration : float
            Kernel duration (seconds). Default: 32.
        device : torch.device, optional
            Device for tensors. None uses default. Pass 'cpu' for CPU testing, 'cuda' for GPU.
        """
        super().__init__()

        self.n_voxels = n_voxels
        self.dt = dt
        self.duration = duration

        # Ensure device is a torch.device object
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._kernels_cache = None  # Cache for pre-computed kernels

        # Select preset
        presets = {
            "spm": SPM_PARAMS,
            "glover": GLOVER_PARAMS,
            "popeye": POPEYE_PARAMS,
            "subcortical": SUBCORTICAL_PARAMS,
        }
        if init_preset.lower() not in presets:
            raise ValueError(
                f"Unknown preset '{init_preset}'. Choose from {list(presets.keys())}"
            )
        preset = presets[init_preset.lower()]

        # Initialise learnable parameters
        self.peak_delay = nn.Parameter(
            torch.full((n_voxels,), preset["peak_delay"], device=self.device, dtype=torch.float32)
        )
        self.peak_disp = nn.Parameter(
            torch.full((n_voxels,), preset["peak_disp"], device=self.device, dtype=torch.float32)
        )
        self.under_delay = nn.Parameter(
            torch.full((n_voxels,), preset["under_delay"], device=self.device, dtype=torch.float32)
        )
        self.under_disp = nn.Parameter(
            torch.full((n_voxels,), preset["under_disp"], device=self.device, dtype=torch.float32)
        )
        self.p_u_ratio = nn.Parameter(
            torch.full((n_voxels,), preset["p_u_ratio"], device=self.device, dtype=torch.float32)
        )

        logger.info("HRFKernel: n_voxels=%d | init_preset='%s' | dt=%.4f s | duration=%.1f s",
                    n_voxels, init_preset, dt, duration)

    def get_kernels(
        self,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build and return per-voxel HRF kernels (no gradients).

        Parameters
        ----------
        indices : torch.Tensor, optional
            Voxel indices to extract kernels for. If None, returns all voxels.
            Shape: (n_query,), values in [0, n_voxels).

        Returns
        -------
        kernels : torch.Tensor, shape (n_query, n_samples)
            Normalised HRF kernels for each requested voxel (detached, no grad).
        """
        with torch.no_grad():
            all_kernels = self._build_kernels_batched()

        if indices is None:
            return all_kernels

        return all_kernels[indices, :]

    def forward(
        self,
        signal: torch.Tensor,
        signal_dt: float = 0.001,
    ) -> torch.Tensor:
        """Convolve per-voxel signals with learnable HRF kernels (GPU-optimized).

        Uses grouped convolution for parallel processing of all voxels.
        Kernels are computed on-the-fly (no caching overhead).

        Parameters
        ----------
        signal : torch.Tensor, shape (n_voxels, n_samples)
            Neural signals (one per voxel).
        signal_dt : float
            Time resolution of signal (seconds). Default: 0.001 (1 ms).

        Returns
        -------
        conv : torch.Tensor, shape (n_voxels, n_samples)
            HRF-convolved signals (same length as input).
        """
        if signal.shape[0] != self.n_voxels:
            raise ValueError(
                f"Signal batch size {signal.shape[0]} != n_voxels {self.n_voxels}"
            )

        n_voxels, n_samples = signal.shape

        # Get kernels: (n_voxels, kernel_len)
        kernels = self._build_kernels_batched()
        kernel_len = kernels.shape[1]

        # Left-pad signal for causal convolution
        # Pad on left (past): (kernel_len - 1) zeros before the signal
        padded_signal = torch.nn.functional.pad(
            signal, (kernel_len - 1, 0), mode="constant", value=0.0
        )  # (n_voxels, n_samples + kernel_len - 1)

        # Reshape for grouped convolution:
        # Conv1d expects: (batch=1, channels=n_voxels, length)
        padded_signal = padded_signal.unsqueeze(0)  # (1, n_voxels, n_samples + kernel_len - 1)
        kernels_conv = kernels.unsqueeze(1)  # (n_voxels, 1, kernel_len) for grouped conv

        # Grouped convolution: each kernel applies to its corresponding voxel independently
        # groups=n_voxels: n_voxels independent convolutions in parallel
        conv = torch.nn.functional.conv1d(
            padded_signal, kernels_conv, groups=n_voxels, padding=0
        )  # (1, n_voxels, n_samples)

        # Remove batch dimension and scale by dt
        conv = conv.squeeze(0) * signal_dt  # (n_voxels, n_samples)

        return conv

    def _build_kernels_batched(self) -> torch.Tensor:
        """Build all HRF kernels in a vectorized manner (GPU-compatible).

        Returns
        -------
        kernels : torch.Tensor, shape (n_voxels, kernel_len)
            Normalised HRF kernels for all voxels.
        """
        t = torch.arange(
            0.0, self.duration, self.dt,
            device=self.device, dtype=torch.float32
        )
        kernel_len = t.shape[0]
        kernels = torch.zeros(
            (self.n_voxels, kernel_len),
            device=self.device, dtype=torch.float32
        )

        # Vectorized gamma PDF computation
        peak_shape = self.peak_delay / self.peak_disp  # (n_voxels,)
        under_shape = self.under_delay / self.under_disp  # (n_voxels,)

        # Expand t for broadcasting: (1, kernel_len) vs (n_voxels, 1)
        t_expanded = t.unsqueeze(0)  # (1, kernel_len)
        peak_shape_exp = peak_shape.unsqueeze(1)  # (n_voxels, 1)
        peak_disp_exp = self.peak_disp.unsqueeze(1)  # (n_voxels, 1)
        under_shape_exp = under_shape.unsqueeze(1)  # (n_voxels, 1)
        under_disp_exp = self.under_disp.unsqueeze(1)  # (n_voxels, 1)
        p_u_ratio_exp = self.p_u_ratio.unsqueeze(1)  # (n_voxels, 1)

        # Compute gamma PDFs with broadcasting
        peak = gamma_pdf_torch(t_expanded, peak_shape_exp, peak_disp_exp)  # (n_voxels, kernel_len)
        under = gamma_pdf_torch(t_expanded, under_shape_exp, under_disp_exp)  # (n_voxels, kernel_len)

        # Double-gamma kernel
        kernel_batch = peak - under / p_u_ratio_exp  # (n_voxels, kernel_len)

        # Normalise each kernel by its positive lobe area
        for v in range(self.n_voxels):
            k = kernel_batch[v, :]
            pos_mask = k > 0
            if pos_mask.any():
                pos_area = torch.sum(k[pos_mask]) * self.dt
                if pos_area > 0:
                    k = k / pos_area
            kernels[v, :] = k

        return kernels
