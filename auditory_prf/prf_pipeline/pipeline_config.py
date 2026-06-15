"""pipeline_config.py
====================
Shared configuration dataclasses for the auditory pRF pipeline.

PipelineConfig  — all per-voxel parameters and execution settings.
ChunkResult     — numpy bridge: stimulus timing outputs from preprocessing,
                  consumed by the torch pipeline in forward().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from auditory_prf.prf_pipeline.hrf_torch import SUBCORTICAL_PARAMS


@dataclass
class PipelineConfig:
    """Per-voxel pipeline configuration.

    Parameters
    ----------
    cf_hz : float
        Initial preferred CF of the target voxel (Hz). Used to initialise
        CFSelector and AdapTransFilter (for Willmore tau init).
    cf_hz_array : np.ndarray, shape (n_cf,)
        Full CF array from the cochlear simulation. Obtain via
        CochleaConfig.get_batch_cf_array() or
        calc_cfs((min_cf, max_cf, n_cf), species='human').
        Required for CFSelector (differentiable CF fitting).
    alpha : float
        Power-law sharpening exponent. Bounded [1, 32].
    pref_dur_ms : float
        Preferred stimulus duration (ms). Bounded [10, 500].
    sigma_dur_ms : float
        Duration tuning width (ms). Bounded [5, 200].
    w : float
        AdapTrans adaptation weight. Bounded (0, 1).
    K : int or None
        AdapTrans kernel length (samples). None = auto (3× max tau).
    tr_s : float
        TR in seconds.
    hrf_params : dict
        HRF preset dict. Default: SUBCORTICAL_PARAMS from hrf_torch.
    noise_model : PmNoise or None
        Pre-instantiated PmNoise object. None = no noise.
    """

    cf_hz: float
    cf_hz_array: Optional[np.ndarray] = field(default=None)
    alpha: float = 2.0
    pref_dur_ms: float = 200.0
    sigma_dur_ms: float = 20.0
    w: float = 0.8
    K: Optional[int] = None
    tr_s: float = 1.0
    hrf_params: dict = field(default_factory=lambda: dict(SUBCORTICAL_PARAMS))
    noise_model: Optional[Any] = None  # PmNoise instance


@dataclass
class ChunkResult:
    """Bridge between numpy preprocessing and the torch gradient graph.

    Produced by chunk_from_id() in chunk_timecourse.py and passed to
    AuditoryPRFPipeline.forward(). Fixed fields (onsets, offsets, tone_dur_ms)
    are not differentiable. Only mean_rates and prf_responses carry gradients.

    Parameters
    ----------
    mean_rates : np.ndarray, shape (n_tones,)
        Per-tone mean firing rate from the sharpened CF timecourse.
        Stored for numpy-pipeline parity checks; recomputed inside forward()
        so gradients flow back to alpha.
    onsets_ms : np.ndarray, shape (n_tones,)
        Bare tone onset times in ms (no chunk margin). Used for boxcar building
        and for recomputing mean rates inside forward().
    offsets_ms : np.ndarray, shape (n_tones,)
        Bare tone offset times in ms. Same use as onsets_ms.
    tone_dur_ms : float
        Duration of each tone in ms (constant within a DIPC sequence).
    total_dur_ms : float
        Total stimulus sequence duration in ms.
    dt_ms : float
        Time resolution in ms. Default 1.0.
    """

    mean_rates: np.ndarray
    onsets_ms: np.ndarray
    offsets_ms: np.ndarray
    tone_dur_ms: float
    total_dur_ms: float
    dt_ms: float = 1.0
    margin_ms: float = 50.0
