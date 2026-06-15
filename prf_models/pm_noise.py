"""BOLD fMRI noise model, ported from gari_pmNoise.m (vistasoft/gari toolbox).

The noise has zero mean. Amplitude parameters (white_amplitude, etc.) are
calibrated against a reference BOLD signal with std == REFERENCE_BOLD_STD
(3% PSC): ``apply_bold_noise()`` scales the generated noise by
``np.std(bold) / REFERENCE_BOLD_STD``, so the 'low'/'mid'/'high' voxel presets
reproduce the SNR levels reported in the prf-Synthesize validation figure
(Lerma-Usabiaga et al.) regardless of the absolute scale of ``bold``. Four
noise components are supported: white noise, cardiac oscillation, respiratory
oscillation, and low-frequency drift (DCT basis, matching SPM's spm_drift).

The parent ``pm`` object must expose:
    pm.TR                 : float  — repetition time (seconds)
    pm.time_points_n      : int    — number of TRs
    pm.time_points_series : ndarray, shape (time_points_n,)  — time axis (s)
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# PmAdapter — minimal shim satisfying the PmNoise 'pm' interface
# ---------------------------------------------------------------------------

@dataclass
class PmAdapter:
    """Minimal shim satisfying the PmNoise ``pm`` interface.

    Parameters
    ----------
    TR : float
        Repetition time in seconds.
    time_points_n : int
        Number of TRs / time points.
    time_points_series : np.ndarray, shape (time_points_n,)
        Time axis in seconds.
    """
    TR: float
    time_points_n: int
    time_points_series: np.ndarray


# ---------------------------------------------------------------------------
# DCT drift basis (mirrors SPM's spm_drift.m)
# ---------------------------------------------------------------------------

def spm_drift(n_timepoints: int, n_basis: int) -> np.ndarray:
    """Discrete cosine drift basis matrix, matching SPM's spm_drift.m exactly.

    Implements SPM's type-II DCT (1-indexed, midpoint sampling):

        C[i, j] = sqrt(2/N) * cos(pi * (2i-1) * (j-1) / (2*N))
        C[:, 0] /= sqrt(2)   # DC column normalisation → 1/sqrt(N)

    This produces orthonormal columns (unit L2 norm), identical to the output
    of MATLAB's ``spm_drift(N, n_basis)``.

    Parameters
    ----------
    n_timepoints : int
        Number of time points (rows), equivalent to ``k`` in SPM notation.
    n_basis : int
        Number of basis functions including the DC (constant) term,
        equivalent to ``N`` in SPM notation.

    Returns
    -------
    np.ndarray, shape (n_timepoints, n_basis)
        Orthonormal columns; column 0 is the DC term (constant, unit norm).
    """
    i = np.arange(1, n_timepoints + 1, dtype=float).reshape(-1, 1)  # 1..N
    j = np.arange(1, n_basis + 1,      dtype=float).reshape(1, -1)  # 1..n_basis
    C = np.sqrt(2.0 / n_timepoints) * np.cos(
        np.pi * (2.0 * i - 1.0) * (j - 1.0) / (2.0 * n_timepoints)
    )
    C[:, 0] /= np.sqrt(2.0)  # DC column: sqrt(2/N) * 1 / sqrt(2) = 1/sqrt(N)
    return C


# ---------------------------------------------------------------------------
# Noise presets (matching MATLAB defaultsGet)
# ---------------------------------------------------------------------------

_VOXEL_DEFAULTS = {
    'mid': dict(
        white_amplitude=0.032,
        cardiac_amplitude=0.01,  cardiac_frequency=1.05,
        respiratory_amplitude=0.01, respiratory_frequency=0.3,
        lowfrequ_amplitude=0.01,  lowfrequ_frequ=120.0,
    ),
    'low': dict(
        white_amplitude=0.016,
        cardiac_amplitude=0.004, cardiac_frequency=1.05,
        respiratory_amplitude=0.004, respiratory_frequency=0.28,
        lowfrequ_amplitude=0.004, lowfrequ_frequ=120.0,
    ),
    'high': dict(
        white_amplitude=0.05,
        cardiac_amplitude=0.025, cardiac_frequency=1.055,
        respiratory_amplitude=0.01,  respiratory_frequency=0.3,
        lowfrequ_amplitude=0.015, lowfrequ_frequ=120.0,
    ),
}

_VOXEL_ALIASES = {
    'mid': 'mid', 'midnoise': 'mid',
    'good': 'low', 'low': 'low', 'lownoise': 'low',
    'bad': 'high', 'high': 'high', 'highnoise': 'high',
}

# BOLD-signal std (as a fraction, i.e. 3% PSC) that the voxel preset amplitudes
# above are calibrated against. apply_bold_noise() rescales the generated
# noise so that np.std(bold) == REFERENCE_BOLD_STD reproduces the published
# SNRs (low=5.29dB, mid=-0.51dB, high=-4.29dB; Lerma-Usabiaga et al. Fig 4).
REFERENCE_BOLD_STD = 0.03


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PmNoise:
    """BOLD fMRI noise model with white, cardiac, respiratory, and drift terms.

    Parameters
    ----------
    pm : object
        Parent pRF model (supplies TR, time_points_n, time_points_series).
    seed : str or int
        'none'/'nonoise' — zero noise; 'random' — random seed each call;
        int — reproducible fixed seed.
    jitter : array_like of length 1–3
        Gaussian SD of perturbations applied to [frequency, amplitude, phase]
        of each periodic component, expressed as a fraction of the nominal
        value.  ``[0, 0]`` means no jitter (default).
    white_amplitude : float
        Std-dev of the zero-mean Gaussian white noise.
    cardiac_amplitude : float
        Peak amplitude of the cardiac sinusoid.
    cardiac_frequency : float
        Cardiac rate (Hz).
    respiratory_amplitude : float
        Peak amplitude of the respiratory sinusoid.
    respiratory_frequency : float
        Breathing rate (Hz).
    lowfrequ_frequ : float
        Period of the low-frequency drift component (seconds).
    lowfrequ_amplitude : float
        Amplitude of the low-frequency drift.
    voxel : str
        Preset noise level used to fill any unspecified amplitude/frequency
        parameters.  One of 'mid' (default), 'low'/'good', 'high'/'bad'.
    """

    @classmethod
    def defaults_get(cls, voxel: str = 'mid') -> dict:
        """Return a copy of the default parameter dict for ``voxel``."""
        key = _VOXEL_ALIASES.get(voxel.lower().replace(' ', ''))
        if key is None:
            raise ValueError(
                f"Voxel type '{voxel}' not recognized. "
                f"Choose from: {list(_VOXEL_ALIASES)}"
            )
        d = {'seed': 'random', 'jitter': np.array([0.0, 0.0])}
        d.update(_VOXEL_DEFAULTS[key])
        return d

    def __init__(
        self,
        pm: Optional[PmAdapter] = None,
        seed: Union[str, int] = 'random',
        jitter: Optional[Union[list, np.ndarray]] = None,
        white_amplitude: Optional[float] = None,
        cardiac_amplitude: Optional[float] = None,
        cardiac_frequency: Optional[float] = None,
        respiratory_amplitude: Optional[float] = None,
        respiratory_frequency: Optional[float] = None,
        lowfrequ_frequ: Optional[float] = None,
        lowfrequ_amplitude: Optional[float] = None,
        voxel: str = 'mid',
    ):
        d = self.defaults_get(voxel)

        self.pm: Optional[PmAdapter] = pm
        self.seed = seed
        self.jitter = np.asarray(
            jitter if jitter is not None else d['jitter'], dtype=float
        )
        self.white_amplitude        = white_amplitude        if white_amplitude        is not None else d['white_amplitude']
        self.cardiac_amplitude      = cardiac_amplitude      if cardiac_amplitude      is not None else d['cardiac_amplitude']
        self.cardiac_frequency      = cardiac_frequency      if cardiac_frequency      is not None else d['cardiac_frequency']
        self.respiratory_amplitude  = respiratory_amplitude  if respiratory_amplitude  is not None else d['respiratory_amplitude']
        self.respiratory_frequency  = respiratory_frequency  if respiratory_frequency  is not None else d['respiratory_frequency']
        self.lowfrequ_frequ         = lowfrequ_frequ         if lowfrequ_frequ         is not None else d['lowfrequ_frequ']
        self.lowfrequ_amplitude     = lowfrequ_amplitude     if lowfrequ_amplitude     is not None else d['lowfrequ_amplitude']
        self.values: Optional[np.ndarray] = None

        self._validate_jitter()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _pm(self) -> PmAdapter:
        if self.pm is None:
            raise RuntimeError(
                "PmNoise.pm is not set — pass a PmAdapter to __init__ or "
                "set noise_model.pm before calling compute()."
            )
        return self.pm

    @property
    def TR(self) -> float:
        return self._pm.TR

    @property
    def values_array(self) -> np.ndarray:
        """Return ``self.values``, raising if ``compute()`` has not been called."""
        if self.values is None:
            raise RuntimeError(
                "PmNoise.values is None — call compute() before accessing the noise array."
            )
        return self.values

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def set_voxel_defaults(self, voxel_type: str) -> None:
        """Reset amplitude/frequency parameters to a built-in noise preset.

        Parameters
        ----------
        voxel_type : str
            One of 'mid', 'low'/'good', 'high'/'bad'.
        """
        d = self.defaults_get(voxel_type)
        for key, val in d.items():
            if key not in ('seed', 'jitter'):
                setattr(self, key, val)

    def compute_white(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Return white noise component, shape (time_points_n,).

        Parameters
        ----------
        rng : numpy Generator, optional
            Shared generator passed by ``compute()`` to thread one RNG through
            all components. If None, a fresh generator is created from
            ``self.seed``.

        Returns
        -------
        np.ndarray, shape (time_points_n,)
        """
        if rng is None:
            rng = self._make_rng()
        if rng is None:
            return np.zeros(self._pm.time_points_n)
        return self._compute_white(rng)

    def compute_cardiac(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Return cardiac sinusoid component, shape (time_points_n,).

        Parameters
        ----------
        rng : numpy Generator, optional
            Shared generator passed by ``compute()``. If None, a fresh
            generator is created from ``self.seed``.

        Returns
        -------
        np.ndarray, shape (time_points_n,)

        Notes
        -----
        When called standalone, the cardiac draw comes from a fresh RNG and
        will differ numerically from the cardiac portion of ``compute()``
        output (where the RNG has already advanced through white noise draws).
        """
        if rng is None:
            rng = self._make_rng()
        if rng is None:
            return np.zeros(self._pm.time_points_n)
        return self._compute_cardiac(rng, self._expand_jitter())

    def compute_respiratory(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Return respiratory sinusoid component, shape (time_points_n,).

        Parameters and return type identical to ``compute_cardiac``.
        """
        if rng is None:
            rng = self._make_rng()
        if rng is None:
            return np.zeros(self._pm.time_points_n)
        return self._compute_respiratory(rng, self._expand_jitter())

    def compute_drift(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Return low-frequency drift component, shape (time_points_n,).

        Parameters and return type identical to ``compute_cardiac``.
        """
        if rng is None:
            rng = self._make_rng()
        if rng is None:
            return np.zeros(self._pm.time_points_n)
        return self._compute_drift(rng, self._expand_jitter())

    def compute(self) -> None:
        """Generate the composite noise signal and store it in ``self.values``.

        Calling this multiple times with ``seed='random'`` produces different
        realisations; a fixed integer seed gives reproducible output.
        """
        rng = self._make_rng()
        if rng is None:
            self.values = np.zeros(self._pm.time_points_n)
            return

        jitter = self._expand_jitter()
        # Draw order is fixed: white → cardiac → respiratory → drift.
        # Private workers consume rng in sequence — preserves reproducibility.
        self.values = (
            self._compute_white(rng)
            + self._compute_cardiac(rng, jitter)
            + self._compute_respiratory(rng, jitter)
            + self._compute_drift(rng, jitter)
        )

    def plot(
        self,
        ax_time=None,
        ax_freq=None,
        color: str = 'b',
        linestyle: str = '-',
    ) -> None:
        """Plot noise in time and frequency domains.

        Parameters
        ----------
        ax_time : matplotlib Axes, optional
            Provide to overlay onto an existing axes; otherwise a new figure
            is created with both time and frequency subplots.
        ax_freq : matplotlib Axes, optional
            Frequency-domain axes (used only when ax_time is also supplied).
        color, linestyle : str
            Passed to ``matplotlib.axes.Axes.plot``.
        """
        self.compute()
        t = self._pm.time_points_series
        n = self._pm.time_points_n

        standalone = ax_time is None
        if standalone:
            fig = plt.figure(figsize=(10, 6))
            ax_time = fig.add_subplot(2, 1, 1)
            ax_freq = fig.add_subplot(2, 1, 2)

        # Time domain
        ax_time.plot(t, self.values_array, color=color, linestyle=linestyle)
        ax_time.grid(True)
        ax_time.set_xlabel('Time (sec)')
        ax_time.set_ylabel('Relative amplitude')
        ax_time.set_title(f'Noise: time domain (not scaled to BOLD), TR={self.TR}')

        # Frequency domain — rfft gives the correct one-sided spectrum including
        # the Nyquist bin (n_fft//2 + 1 points total for even n_fft)
        n_fft = n if n % 2 == 0 else n - 1
        y = self.values_array[:n_fft]
        F = np.abs(np.fft.rfft(y)) / n_fft
        F[1:-1] *= 2  # double interior bins; leave DC and Nyquist as-is
        freqs = np.fft.rfftfreq(n_fft, d=self.TR)

        if ax_freq is not None:
            ax_freq.plot(freqs, F, color=color, linestyle=linestyle)
            ax_freq.grid(True)
            ax_freq.set_xlabel('Frequency (Hz)')
            ax_freq.set_ylabel('Relative amplitude')
            ax_freq.set_title(
                f'Noise: frequency domain (not scaled to BOLD), TR={self.TR}'
            )

        if standalone:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_jitter(self) -> None:
        j = self.jitter
        if j.ndim != 1 or not (1 <= len(j) <= 3):
            raise ValueError("jitter must be a numeric vector of length 1, 2, or 3")
        if not (0.0 <= j[0] <= 1.0):
            raise ValueError("jitter[0] (frequency jitter) must be in [0, 1]")
        if len(j) >= 2 and j[1] < 0:
            raise ValueError("jitter[1] (amplitude jitter) must be >= 0")
        if len(j) >= 3 and j[2] < 0:
            raise ValueError("jitter[2] (phase jitter) must be >= 0")

    def _expand_jitter(self) -> np.ndarray:
        j = self.jitter
        if len(j) == 1:
            warnings.warn(
                "Only one jitter value provided; applied to freq and amplitude; "
                "phase jitter set to 0."
            )
            return np.array([j[0], j[0], 0.0])
        elif len(j) == 2:
            return np.array([j[0], j[1], 0.0])
        return j

    def _make_rng(self):
        """Return a numpy RandomState, or None for the no-noise case.

        Uses ``np.random.RandomState(seed)``, which initialises MT19937 via
        ``init_genrand`` — the same C-level seeding path as MATLAB's
        ``rng(seed, 'twister')``.  Confirmed compatible: both produce the
        identical uniform stream from the same integer seed.

        Ceiling on MATLAB agreement
        ---------------------------
        The uniform stream is identical to MATLAB's.  The normal stream
        (``randn``) is NOT: MATLAB and NumPy each ship their own Marsaglia-
        Tsang ziggurat tables, so ``standard_normal()`` draws diverge even
        though the underlying MT state is the same.  This is the hard limit
        of Python/MATLAB compatibility without reimplementing MATLAB's ziggurat.

        Components affected:
          - White noise (vector of normals)          — cannot match sample-by-sample
          - Cardiac/respiratory jitter (3 normals each) — cannot match sample-by-sample
          - Drift jitter (2 normals)                 — cannot match sample-by-sample
          - Drift signal (spm_drift, deterministic)  — exact match ✓
        """
        seed = self.seed
        if isinstance(seed, str) and seed.lower() in ('none', 'nonoise'):
            return None
        if isinstance(seed, str) and seed.lower() == 'random':
            return np.random.RandomState()
        if isinstance(seed, (int, np.integer)):
            return np.random.RandomState(int(seed))
        raise ValueError(f"Unknown seed value: {seed!r}")

    def _compute_white(self, rng: np.random.RandomState) -> np.ndarray:
        """White noise draw. Consumes time_points_n draws from rng iff amplitude > 0."""
        n = self._pm.time_points_n
        if self.white_amplitude <= 0:
            return np.zeros(n)
        return self.white_amplitude * rng.standard_normal(n)

    def _compute_cardiac(self, rng: np.random.RandomState, jitter: np.ndarray) -> np.ndarray:
        """Cardiac sinusoid. Consumes 3 scalar draws from rng iff amplitude > 0."""
        n = self._pm.time_points_n
        if self.cardiac_amplitude <= 0:
            return np.zeros(n)
        f = self.cardiac_frequency * (1 + jitter[0] * rng.standard_normal())
        a = self.cardiac_amplitude * (1 + jitter[1] * rng.standard_normal())
        p = 2 * np.pi * jitter[2]  *      rng.standard_normal()
        return a * np.sin(2 * np.pi * self._pm.time_points_series * f + p)

    def _compute_respiratory(self, rng: np.random.RandomState, jitter: np.ndarray) -> np.ndarray:
        """Respiratory sinusoid. Consumes 3 scalar draws from rng iff amplitude > 0."""
        n = self._pm.time_points_n
        if self.respiratory_amplitude <= 0:
            return np.zeros(n)
        f = self.respiratory_frequency * (1 + jitter[0] * rng.standard_normal())
        a = self.respiratory_amplitude * (1 + jitter[1] * rng.standard_normal())
        p = 2 * np.pi * jitter[2]     *      rng.standard_normal()
        return a * np.sin(2 * np.pi * self._pm.time_points_series * f + p)

    def _compute_drift(self, rng: np.random.RandomState, jitter: np.ndarray) -> np.ndarray:
        """Low-frequency drift. Consumes 2 scalar draws from rng iff amplitude > 0."""
        n = self._pm.time_points_n
        if self.lowfrequ_amplitude <= 0:
            return np.zeros(n)
        f_noise = self.lowfrequ_frequ     * (1 + jitter[0] * rng.standard_normal())
        a_noise = self.lowfrequ_amplitude * (1 + jitter[1] * rng.standard_normal())
        n_basis = int(np.floor(2 * (n * self.TR) / f_noise + 1))
        if n_basis < 3:
            raise ValueError(
                "Drift basis requires at least 3 functions. "
                "Reduce lowfrequ_frequ or increase scan duration."
            )
        drift_signal = spm_drift(n, n_basis)[:, 1:].sum(axis=1)
        return a_noise * drift_signal


# ---------------------------------------------------------------------------
# Functional helper for pipeline integration
# ---------------------------------------------------------------------------

def apply_bold_noise(bold: np.ndarray, noise_model: PmNoise, tr_s: float) -> np.ndarray:
    """Add physiological and acquisition noise to a clean BOLD timeseries.

    Noise amplitude is SNR-based: the raw noise realisation from
    ``noise_model.compute()`` (in fractional units, e.g. 0.032 for 'mid'
    white noise) is scaled by ``np.std(bold) / REFERENCE_BOLD_STD`` before
    being added, so the 'low'/'mid'/'high' voxel presets reproduce their
    target SNRs regardless of the absolute scale of ``bold``.

    Parameters
    ----------
    bold : np.ndarray, shape (n_tr,)
        Clean BOLD prediction from the pipeline.
    noise_model : PmNoise
        Configured noise model. ``compute()`` is called internally; the caller
        controls reproducibility via ``noise_model.seed``.
    tr_s : float
        Repetition time in seconds (must match the TR used to produce ``bold``).

    Returns
    -------
    np.ndarray, shape (n_tr,)
        ``bold`` plus the generated noise realisation, scaled by
        ``np.std(bold) / REFERENCE_BOLD_STD``.
    """
    n_tr = len(bold)
    noise_model.pm = PmAdapter(
        TR=tr_s,
        time_points_n=n_tr,
        time_points_series=np.arange(n_tr, dtype=float) * tr_s,
    )
    noise_model.compute()
    signal_scale = np.std(bold) / REFERENCE_BOLD_STD
    return bold + noise_model.values_array * signal_scale


if __name__ == '__main__':
    TR = 1.6
    n_tr = 300
    pm = PmAdapter(
        TR=TR,
        time_points_n=n_tr,
        time_points_series=np.arange(n_tr, dtype=float) * TR,
    )

    levels = [('low', 'steelblue'), ('mid', 'darkorange'), ('high', 'firebrick')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col')
    fig.suptitle('BOLD noise model — all voxel levels (seed=42)', fontsize=13)

    for row, (voxel, color) in enumerate(levels):
        noise = PmNoise(pm=pm, seed=42, voxel=voxel)
        noise.plot(ax_time=axes[row, 0], ax_freq=axes[row, 1], color=color)
        axes[row, 0].set_title(f'{voxel}  —  time domain')
        axes[row, 1].set_title(f'{voxel}  —  frequency domain')

    plt.tight_layout()
    plt.show()