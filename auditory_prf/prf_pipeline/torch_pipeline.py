"""torch_pipeline.py
====================
PyTorch nn.Module pipeline for gradient-based auditory pRF fitting.

Each pipeline stage is a separate nn.Module so requires_grad_ can be toggled
independently per model variant. Existing numpy functions are kept unchanged;
these modules re-implement their core logic in torch for differentiability.

Gradient boundary: cochlear .npz files are loaded as numpy, converted to
torch.Tensor at the start of forward(). The Zilany2014 cochlear model (Cython)
stays as offline preprocessing.

Modules
-------
CFSelector          — cf_x ∈ (x_min, x_max) Greenwood space; always learnable
PowerLawSharpening  — alpha ∈ [1, 32]
DurationFilter      — pref_dur_ms ∈ [10, 500], sigma_dur_ms ∈ [5, 200]
AdapTransFilter     — tau_on_ms, tau_off_ms ∈ [10, 500], w ∈ (0,1), on_off_ratio ∈ (0,1)
HRFConvolution      — fixed kernel (register_buffer)
NoiseModel          — wraps PmNoise, no grad
AuditoryPRFPipeline — composes all stages; freeze/unfreeze via model_variant

Helper functions
----------------
recompute_mean_rates  — differentiable per-tone mean rate from sharpened timecourse
build_boxcar_torch    — differentiable boxcar train via mask matrix matmul
"""

from __future__ import annotations

import math
import sys
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auditory_prf.prf_pipeline.adaptrans_onoff_filters import tau_to_a
from auditory_prf.prf_pipeline.hrf_torch import (
    SUBCORTICAL_PARAMS,
    build_hrf_kernel_torch,
    convolve_hrf_torch_causal,
)
from auditory_prf.prf_pipeline.pipeline_config import ChunkResult, PipelineConfig


# ── Helper functions ───────────────────────────────────────────────────────────

def recompute_mean_rates(
    cf_timecourse: torch.Tensor,
    chunk: ChunkResult,
) -> torch.Tensor:
    """Compute per-tone mean firing rates inside the torch graph.

    Must be called inside forward() — NOT pre-computed from numpy — so that
    gradients flow back through cf_timecourse to PowerLawSharpening.alpha.

    Uses onset → (offset + margin_ms) windows, matching chunk_timecourse.py.
    The margin captures post-tone neural decay; critical for the OFF path where
    the peak response occurs shortly after tone offset.

    Parameters
    ----------
    cf_timecourse : Tensor, shape (n_time,)
        Sharpened single-CF timecourse.
    chunk : ChunkResult
        Stimulus timing from preprocessing.

    Returns
    -------
    mean_rates : Tensor, shape (n_tones,)
    """
    rates = []
    T = cf_timecourse.shape[0]
    for on_ms, off_ms in zip(chunk.onsets_ms, chunk.offsets_ms):
        i_on  = max(0, round(float(on_ms)  / chunk.dt_ms))
        i_off = min(T, round(float(off_ms + chunk.margin_ms) / chunk.dt_ms))
        window = cf_timecourse[i_on:i_off]
        if window.numel() > 0:
            rates.append(window.mean())
        else:
            rates.append(cf_timecourse.new_zeros(()))
    return torch.stack(rates)  # (n_tones,)


def build_boxcar_torch(
    prf_responses: torch.Tensor,
    chunk: ChunkResult,
) -> torch.Tensor:
    """Build a differentiable 1ms-resolution boxcar amplitude train.

    Uses a fixed mask matrix (n_tones × n_samples) so that a single matmul
    carries gradients from prf_responses to every sample in the output train.
    The mask itself is not differentiable (onsets/offsets are fixed).

    Parameters
    ----------
    prf_responses : Tensor, shape (n_tones,)
    chunk : ChunkResult

    Returns
    -------
    train : Tensor, shape (n_samples,)
    """
    n_samples = math.ceil(chunk.total_dur_ms / chunk.dt_ms)
    n_tones   = len(chunk.onsets_ms)

    masks = prf_responses.new_zeros(n_tones, n_samples)
    for s, (on_ms, off_ms) in enumerate(zip(chunk.onsets_ms, chunk.offsets_ms)):
        i_on  = max(0, min(round(float(on_ms)  / chunk.dt_ms), n_samples))
        i_off = max(0, min(round(float(off_ms) / chunk.dt_ms), n_samples))
        if i_on < i_off:
            masks[s, i_on:i_off] = 1.0

    return prf_responses @ masks  # (n_samples,)


# ── Pipeline stages ────────────────────────────────────────────────────────────

class CFSelector(nn.Module):
    """Differentiable CF selection via linear interpolation in Greenwood x-space.

    CFs from calc_cfs() are evenly spaced in Greenwood position x, so
    interpolating in x-space gives uniform gradient sensitivity across the
    frequency range. Interpolating in Hz would give ~20× larger gradient steps
    near 2500 Hz than near 125 Hz, making optimization unstable.

    Because x_map is evenly spaced, the fractional row index is computed
    directly as (cf_x - x_min) / x_step — no search needed.

    Greenwood constants (A=165.4, k=0.88, a=2.1) match calc_cfs() in
    cochlea/cochlea/zilany2014/util.py (human species).

    Parameters
    ----------
    cf_hz_array : np.ndarray, shape (n_cf,)
        CF values in Hz from the cochlear simulation. Obtain via
        CochleaConfig.get_batch_cf_array() or
        calc_cfs((min_cf, max_cf, n_cf), species='human').
    init_cf_hz : float
        Initial preferred CF in Hz. Clamped to [cf_hz_array[0], cf_hz_array[-1]].
    """

    _A: float = 165.4   # Greenwood A (Hz)
    _k: float = 0.88    # Greenwood k
    _a: float = 2.1     # Greenwood a (position units)

    def __init__(self, cf_hz_array: np.ndarray, init_cf_hz: float) -> None:
        super().__init__()
        cf_hz_array = np.asarray(cf_hz_array, dtype=np.float64)
        n_cf = len(cf_hz_array)

        # Convert to Greenwood x-space (evenly spaced by construction of calc_cfs)
        x_map = np.log10(cf_hz_array / self._A + self._k) / self._a  # (n_cf,)
        self.register_buffer('x_map', torch.from_numpy(x_map.astype(np.float32)))
        self.n_cf   = n_cf
        self.x_min  = float(x_map[0])
        self.x_max  = float(x_map[-1])
        self.x_step = (self.x_max - self.x_min) / (n_cf - 1)

        # Parameterise in [0,1] via logit so optimizer is unconstrained
        init_hz = float(np.clip(init_cf_hz, float(cf_hz_array[0]), float(cf_hz_array[-1])))
        init_x  = math.log10(init_hz / self._A + self._k) / self._a
        init_t  = float(np.clip((init_x - self.x_min) / (self.x_max - self.x_min), 1e-6, 1.0 - 1e-6))
        self._logit_x = nn.Parameter(torch.tensor(math.log(init_t / (1.0 - init_t))))

    @property
    def cf_x(self) -> torch.Tensor:
        """Current CF as Greenwood position x (bounded to [x_min, x_max])."""
        return torch.sigmoid(self._logit_x) * (self.x_max - self.x_min) + self.x_min

    @property
    def cf_hz(self) -> torch.Tensor:
        """Current CF in Hz (for logging/inspection)."""
        return self._A * (10.0 ** (self._a * self.cf_x) - self._k)

    def _interp(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Linear interpolation at the selected CF. spectrum: (n_cf, T) → (T,)."""
        frac_idx = (self.cf_x - self.x_min) / self.x_step   # continuous ∈ [0, n_cf-1]
        below = frac_idx.long().clamp(0, self.n_cf - 2)      # integer index, no grad
        t = (frac_idx - below.float()).clamp(0.0, 1.0)       # weight ∈ [0,1], grad flows
        return (1.0 - t) * spectrum[below] + t * spectrum[below + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CF timecourse(s) via linear interpolation in Greenwood x-space.

        Parameters
        ----------
        x : Tensor, shape (n_cf, T) or (2, n_cf, T)
            Population PSTH or AdapTrans ON/OFF output.

        Returns
        -------
        Tensor, shape (T,) if input is (n_cf, T),
                      (2, T) if input is (2, n_cf, T).
        """
        if x.ndim == 3:
            # (2, n_cf, T) — apply to both ON and OFF channels
            return torch.stack([self._interp(x[0]), self._interp(x[1])], dim=0)
        return self._interp(x)   # (n_cf, T) → (T,)


class PowerLawSharpening(nn.Module):
    """Power-law sharpening of cochlear population PSTH.

    alpha stored in log-space (_log_alpha) so the optimizer can move freely
    while exp() always yields a positive value. Hard-clamped to [ALPHA_MIN, ALPHA_MAX].

    Parameters
    ----------
    alpha : float
        Initial sharpening exponent. Default 2.0.
    """

    ALPHA_MIN: float = 1.0
    ALPHA_MAX: float = 32.0

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        alpha_init = float(np.clip(alpha, self.ALPHA_MIN, self.ALPHA_MAX))
        self._log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self._log_alpha).clamp(self.ALPHA_MIN, self.ALPHA_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, any shape
            Firing rate values to sharpen (e.g. (n_tones,) mean rates).

        Returns
        -------
        sharpened : Tensor, same shape as input
            Mean-normalized: mean(sharpened) == mean(input).
        """
        alpha     = self.alpha
        sharpened = x ** alpha
        in_mean   = x.mean()
        out_mean  = sharpened.mean()
        if out_mean > 0:
            sharpened = sharpened * (in_mean / out_mean)
        return sharpened


class DurationFilter(nn.Module):
    """Gaussian duration tuning filter.

    Scales each tone's mean firing rate by a Gaussian centred on pref_dur_ms.
    Both parameters stored in log-space for positivity.

    Parameters
    ----------
    pref_dur_ms : float
        Preferred stimulus duration in ms. Bounded [10, 500].
    sigma_dur_ms : float
        Duration tuning width in ms. Bounded [5, 200].
    """

    PREF_DUR_MIN_MS:  float = 10.0
    PREF_DUR_MAX_MS:  float = 500.0
    SIGMA_DUR_MIN_MS: float = 5.0
    SIGMA_DUR_MAX_MS: float = 200.0

    def __init__(self, pref_dur_ms: float = 200.0, sigma_dur_ms: float = 20.0) -> None:
        super().__init__()
        pref  = float(np.clip(pref_dur_ms,  self.PREF_DUR_MIN_MS,  self.PREF_DUR_MAX_MS))
        sigma = float(np.clip(sigma_dur_ms, self.SIGMA_DUR_MIN_MS, self.SIGMA_DUR_MAX_MS))
        self._log_pref_dur  = nn.Parameter(torch.tensor(math.log(pref)))
        self._log_sigma_dur = nn.Parameter(torch.tensor(math.log(sigma)))

    @property
    def pref_dur_ms(self) -> torch.Tensor:
        return torch.exp(self._log_pref_dur).clamp(self.PREF_DUR_MIN_MS, self.PREF_DUR_MAX_MS)

    @property
    def sigma_dur_ms(self) -> torch.Tensor:
        return torch.exp(self._log_sigma_dur).clamp(self.SIGMA_DUR_MIN_MS, self.SIGMA_DUR_MAX_MS)

    def forward(self, mean_rates: torch.Tensor, tone_dur_ms: float) -> torch.Tensor:
        """
        Parameters
        ----------
        mean_rates : Tensor, shape (n_tones,)
        tone_dur_ms : float
            Duration of each tone in ms (constant within a DIPC sequence).

        Returns
        -------
        prf_responses : Tensor, shape (n_tones,)
        """
        pref  = self.pref_dur_ms
        sigma = self.sigma_dur_ms
        gauss = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * \
                torch.exp(-0.5 * ((tone_dur_ms - pref) / sigma) ** 2)
        return mean_rates * gauss


class AdapTransFilter(nn.Module):
    """AdapTrans FIR onset/offset filter applied to the full population PSTH.

    Matches the deepSTRF AdapTrans architecture: vectorized grouped conv1d across
    all CF channels, with the d/p parameterisation from Rançon et al. (2024):
        a = 1 / (1 + d²)   d ∈ ℝ → a ∈ (0, 1)
        w = 1 / (1 + p²)   p ∈ ℝ → w ∈ (0, 1)
    Kernels are pre-flipped during building so F.conv1d (cross-correlation) produces
    the correct causal convolution result.

    Per-CF parameters (n_cf,): d_on, d_off initialised from a fixed init_tau_ms;
    p initialised uniformly from w. tau is a free parameter — no CF-tau relationship
    is assumed.

    Parameters
    ----------
    cf_hz_array : np.ndarray, shape (n_cf,)
        CF array. Pass a single-element array for the 1-D boxcar use case.
    w : float
        Initial adaptation weight for all CFs. Default 0.8.
    K : int or None
        Kernel length (samples). None = auto (3 × init_tau_ms + 1).
    init_tau_ms : float
        Initial time constant (ms) for all CFs. Default 100.0.
    """

    def __init__(
        self, cf_hz_array: np.ndarray, w: float = 0.8, K: Optional[int] = None,
        init_tau_ms: float = 100.0,
    ) -> None:
        super().__init__()
        cf_hz_array = np.asarray(cf_hz_array, dtype=np.float64)
        self.n_cf = len(cf_hz_array)

        # Uniform init tau → a → d for all CFs (no CF-tau relationship)
        a_init = float(tau_to_a(init_tau_ms, dt_ms=1.0))
        d_init = float(math.sqrt(1.0 / a_init - 1.0))

        # p = sqrt(1/w - 1): maps w ∈ (0,1) → p ∈ (0,∞)
        w_clip = float(np.clip(w, 1e-4, 1.0 - 1e-4))
        p_init = float(math.sqrt(1.0 / w_clip - 1.0))

        self.d_on  = nn.Parameter(torch.full((self.n_cf,), d_init))
        self.d_off = nn.Parameter(torch.full((self.n_cf,), d_init))
        self.p     = nn.Parameter(torch.full((self.n_cf,), p_init))

        if K is None:
            K = int(round(3.0 * init_tau_ms)) + 1
        self.K = K

    def _build_on_kernels(self, d: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Build ON kernels for all CFs. Returns (n_cf, 1, K), pre-flipped.

        Matches deepSTRF AdapTrans.ON_kernel() exactly.
        """
        K      = self.K
        device = d.device
        a = 1.0 / (1.0 + d ** 2)                               # (n_cf,) ∈ (0,1)
        w = 1.0 / (1.0 + p ** 2)                               # (n_cf,) ∈ (0,1)
        ones = torch.ones(K - 1, device=device, dtype=d.dtype)
        rng  = torch.arange(K - 1, device=device, dtype=d.dtype)

        # C: normalization per CF (n_cf,)
        C = 1.0 / (torch.outer(a, ones) ** rng).sum(dim=1)

        # Build (n_cf, 1, K): tap[0]=1, taps[1:]=-C*w*a^rng
        kernel = torch.ones(self.n_cf, 1, K, device=device, dtype=d.dtype)
        kernel[:, 0, 1:] = (-C * w).unsqueeze(-1) * (torch.outer(a, ones) ** rng)

        # Pre-flip: after flip, tap[-1]=1 (current); used directly with F.conv1d
        return torch.flip(kernel, dims=(2,))

    def _build_off_kernels(self, d: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Build OFF kernels for all CFs. Returns (n_cf, 1, K), pre-flipped.

        Matches deepSTRF AdapTrans.OFF_kernel() exactly.
        After pre-flip the 'current sample' tap is at index [-1]; it is corrected
        from -1/w to -w using torch.cat (avoids in-place ops on the autograd graph).
        """
        kernel_on = self._build_on_kernels(d, p)                # (n_cf, 1, K)
        w = (1.0 / (1.0 + p ** 2)).view(-1, 1, 1)              # (n_cf, 1, 1)
        kernel_off_base = -kernel_on / w                        # (n_cf, 1, K)

        # Replace last tap: -1/w → -w (current-sample correction, no in-place)
        w_vec    = (1.0 / (1.0 + p ** 2)).view(-1, 1, 1)       # (n_cf, 1, 1)
        last_tap = -w_vec                                       # (n_cf, 1, 1) = -w
        return torch.cat([kernel_off_base[:, :, :-1], last_tap], dim=2)  # (n_cf, 1, K)

    def forward(self, sharpened: torch.Tensor) -> torch.Tensor:
        """Apply ON and OFF AdapTrans filters to all CFs simultaneously.

        Uses grouped F.conv1d (one kernel per CF). Padding uses 'replicate' mode
        to avoid suppressing the first onset in each CF channel.

        Parameters
        ----------
        sharpened : Tensor, shape (n_cf, T)
            Power-law sharpened population PSTH at 1ms resolution.

        Returns
        -------
        adapted : Tensor, shape (2, n_cf, T)
            adapted[0] = ON responses, adapted[1] = OFF responses.
        """
        n_cf, T = sharpened.shape
        K = self.K

        kernel_on  = self._build_on_kernels( self.d_on,  self.p)   # (n_cf, 1, K)
        kernel_off = self._build_off_kernels(self.d_off, self.p)   # (n_cf, 1, K)

        # Grouped conv1d: each CF gets its own kernel
        signal = sharpened.unsqueeze(0)                             # (1, n_cf, T)
        padded = F.pad(signal, (K - 1, 0), mode='replicate')       # (1, n_cf, T+K-1)

        out_on  = F.conv1d(padded, kernel_on,  stride=1, groups=n_cf)  # (1, n_cf, T)
        out_off = F.conv1d(padded, kernel_off, stride=1, groups=n_cf)  # (1, n_cf, T)

        return torch.stack(
            [out_on.squeeze(0), out_off.squeeze(0)], dim=0
        )  # (2, n_cf, T)


class HRFConvolution(nn.Module):
    """Fixed double-gamma HRF convolution with TR downsampling.

    Kernel is pre-built from an HRF preset dict and stored as a buffer —
    it is not learnable and moves to GPU automatically with .to(device).

    Parameters
    ----------
    hrf_params : dict
        Keys: peak_delay, peak_disp, under_delay, under_disp, p_u_ratio.
    tr_s : float
        TR in seconds (for downsampling).
    signal_dt_s : float
        Signal time resolution in seconds. Default 0.001 (1ms).
    """

    def __init__(
        self, hrf_params: dict, tr_s: float, signal_dt_s: float = 0.001
    ) -> None:
        super().__init__()
        self.tr_s         = tr_s
        self.signal_dt_s  = signal_dt_s
        self._downsample  = round(tr_s / signal_dt_s)

        kernel, _ = build_hrf_kernel_torch(**hrf_params, dt=signal_dt_s)
        self.register_buffer('kernel', kernel)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Convolve signal with HRF and downsample to TR.

        Parameters
        ----------
        signal : Tensor, shape (T,)
            1ms-resolution neural response.

        Returns
        -------
        bold : Tensor, shape (n_tr,)
        """
        convolved = convolve_hrf_torch_causal(signal, self.kernel, self.signal_dt_s)
        return convolved[::self._downsample]


class NoiseModel(nn.Module):
    """BOLD noise injection using PmNoise.

    Not learnable — used for forward-model simulation only.
    The PmNoise object must be pre-instantiated and passed in; the adapter
    (time axis) is re-set per forward() call to match the actual BOLD length.

    Parameters
    ----------
    pm_noise : PmNoise
        Pre-instantiated noise model from prf_models/pm_noise.py.
    tr_s : float
        TR in seconds (used to build the time axis for PmAdapter).
    """

    def __init__(self, pm_noise: object, tr_s: float) -> None:
        super().__init__()
        self._pm_noise = pm_noise
        self.tr_s      = tr_s

    def forward(self, bold: torch.Tensor) -> torch.Tensor:
        """Add noise to BOLD signal.

        Parameters
        ----------
        bold : Tensor, shape (n_tr,)

        Returns
        -------
        noisy_bold : Tensor, shape (n_tr,)
        """
        # Import here to avoid hard dependency when noise is unused
        _root = os.path.join(os.path.dirname(__file__), '..', '..', 'prf_models')
        if _root not in sys.path:
            sys.path.insert(0, os.path.abspath(_root))
        from pm_noise import PmAdapter, REFERENCE_BOLD_STD  # noqa: PLC0415

        n_tr = bold.shape[0]
        self._pm_noise.pm = PmAdapter(
            TR=self.tr_s,
            time_points_n=n_tr,
            time_points_series=np.arange(n_tr, dtype=np.float32) * self.tr_s,
        )
        self._pm_noise.compute()
        noise = torch.from_numpy(
            self._pm_noise.values_array.astype(np.float32)
        ).to(bold.device)
        # SNR-based scaling, matching apply_bold_noise(); detached so the
        # noise magnitude doesn't add a second gradient path through bold.
        signal_scale = bold.detach().std() / REFERENCE_BOLD_STD
        return bold + noise * signal_scale


# ── Full pipeline ──────────────────────────────────────────────────────────────

class AuditoryPRFPipeline(nn.Module):
    """Full auditory pRF forward model pipeline (deepSTRF-style AdapTrans).

    Pipeline order:
      CFSelector → [mean rates] → PowerLawSharpening → DurationFilter →
      BoxcarBuilder → AdapTransFilter (1-D boxcar) → HRFConvolution →
      on_off_ratio mix → NoiseModel (optional)

    AdapTrans is applied to the flat mean-rate boxcar after CF selection,
    matching the numpy pipeline semantics. on_off_ratio is a standalone
    parameter on this module (moved out of AdapTransFilter).
    AdapTransFilter is initialised for the single starting CF (not the full
    population); d_on, d_off, p are shape (1,) and jointly optimised with CF.

    Model variants:
      variant 0 — alpha + CF only; no AdapTrans, no duration filter
                  (matches full_pipeline_notemporal.py exactly)
      variant 1 — alpha + CF only (AdapTrans + duration applied but frozen)
      variant 2 — alpha + CF + duration (pref_dur_ms, sigma_dur_ms)
      variant 3 — alpha + CF + adaptrans (d_on, d_off, p) + on_off_ratio
      variant 4 — all stages free (default)

    Parameters
    ----------
    config : PipelineConfig
    model_variant : int
        1–4, controls which parameters have requires_grad=True.
    """

    def __init__(self, config: PipelineConfig, model_variant: int = 4) -> None:
        super().__init__()
        self.config        = config
        self.model_variant = model_variant
        self.sharpening  = PowerLawSharpening(config.alpha)
        self.adaptrans   = AdapTransFilter(np.array([config.cf_hz]), config.w, config.K)
        self.cf_selector = CFSelector(config.cf_hz_array, config.cf_hz)
        self.duration    = DurationFilter(config.pref_dur_ms, config.sigma_dur_ms)
        self.hrf         = HRFConvolution(config.hrf_params, config.tr_s)
        self._logit_on_off_ratio = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5
        self.noise       = (
            NoiseModel(config.noise_model, config.tr_s)
            if config.noise_model is not None else None
        )
        self._apply_variant_freezing(model_variant)

    @property
    def on_off_ratio(self) -> torch.Tensor:
        return torch.sigmoid(self._logit_on_off_ratio)

    def _apply_variant_freezing(self, variant: int) -> None:
        """Freeze stages not active in this model variant."""
        self.duration.requires_grad_(variant in (2, 4))
        self.adaptrans.requires_grad_(variant in (3, 4))
        self._logit_on_off_ratio.requires_grad_(variant in (3, 4))
        # variant 0: AdapTrans and duration are both skipped in forward(), not just frozen
        if variant == 0:
            self.adaptrans.requires_grad_(False)
            self.duration.requires_grad_(False)
            self._logit_on_off_ratio.requires_grad_(False)

    def forward(
        self,
        population_psth: torch.Tensor,
        chunk: ChunkResult,
    ) -> torch.Tensor:
        """Run forward model from cochlear PSTH to BOLD.

        Parameters
        ----------
        population_psth : Tensor, shape (n_cf, n_time)
            Loaded from .npz via torch.from_numpy(). Gradient boundary.
        chunk : ChunkResult
            Pre-computed stimulus timing from numpy preprocessing.

        Returns
        -------
        bold : Tensor, shape (n_tr,)
            Predicted BOLD signal at TR resolution.
        """
        # Stage 1: CF selection on raw PSTH (gradient path to cf_x)
        cf_tc = self.cf_selector(population_psth)             # (T,)

        # Stage 2: per-tone mean rates (recomputed inside graph for alpha grad)
        mean_rates = recompute_mean_rates(cf_tc, chunk)       # (n_tones,)

        # Stage 3: power-law sharpening on mean rates
        sharpened = self.sharpening(mean_rates)               # (n_tones,)

        if self.model_variant == 0:
            # Notemporal path: no AdapTrans, no duration filter.
            # Matches full_pipeline_notemporal.py exactly.
            train = build_boxcar_torch(sharpened, chunk)      # (T_total,)
            bold  = self.hrf(train)                           # (n_tr,)
        else:
            # Stage 4: duration filter
            prf = self.duration(sharpened, chunk.tone_dur_ms) # (n_tones,)

            # Stage 5: flat boxcar from pRF-weighted mean rates
            train = build_boxcar_torch(prf, chunk)            # (T_total,)

            # Stage 6: AdapTrans ON/OFF on flat boxcar (single-CF filter)
            on_off  = self.adaptrans(train.unsqueeze(0))      # (2, 1, T_total)
            on_tc   = on_off[0, 0]                            # (T_total,)
            off_tc  = on_off[1, 0]                            # (T_total,)

            # Stage 7: HRF + ON/OFF mix
            bold_on  = self.hrf(on_tc)                        # (n_tr,)
            bold_off = self.hrf(off_tc)                       # (n_tr,)
            ratio    = self.on_off_ratio
            bold     = ratio * bold_on + (1.0 - ratio) * bold_off

        # Stage 7: noise (optional, not differentiable)
        if self.noise is not None:
            with torch.no_grad():
                bold = self.noise(bold)

        return bold
