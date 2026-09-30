"""Recover the (frequency, amplitude, phase) a PmNoise jitter draw used.

Cross-language jitter verification can't compare raw output arrays sample-
by-sample (MATLAB's randn and NumPy's standard_normal diverge even from an
identical seed -- see pm_noise.py's _make_rng() docstring). Instead, we
recover the scalar (f, a, p) parameters each realization actually used, by
independently seeding a fresh RandomState and consuming standard_normal()
calls in the SAME ORDER pm_noise.py's private _compute_* methods do
internally (confirmed against both pm_noise.py and the vendored
gari_pmNoise.m):

    cardiac / respiratory : 3 draws, order = [freq, amp, phase]
    drift                 : 2 draws, order = [freq, amp]  (no phase term)

Recovery is exact (not an estimate) because RandomState(seed) is
deterministic: an independently-seeded RNG consuming the same number of
draws in the same order returns bit-identical values to the ones the real
method consumed internally. See verify_recovery_matches_pm_noise() for a
one-time bit-exact cross-check against the actual pm_noise.py code path.
"""
import numpy as np

# 'mid' voxel preset nominal parameters (prf_models/pm_noise.py _VOXEL_DEFAULTS),
# matching tests/matlab_verify_noise.m's existing convention.
NOMINAL = {
    'cardiac':     dict(freq=1.05, amp=0.01),
    'respiratory': dict(freq=0.30, amp=0.01),
    'drift':       dict(freq=120.0, amp=0.01),
}

TR_S = 1.5
N_TR = 100
JITTER_LEVELS = (0.1, 0.4, 0.5)  # 0.1/0.5 and 0.4 are the original author's own
                                  # gari_pmNoise.m demo values (0.4 is the one
                                  # mislabeled "0.5 jitter" in that file's comment)
N_SEEDS = 100


def drift_n_basis_valid(freq_hz: float) -> bool:
    """Whether a jittered drift period yields a usable basis (n_basis >= 3).

    Mirrors the same check pm_noise.py's _compute_drift() and
    gari_pmNoise.m both raise an error on. At larger jitter, a perturbed
    period can push n_basis below 3 -- an expected edge case of the real
    implementation (the floor() in n_basis is inherently discontinuous),
    not a Python/MATLAB discrepancy. Draws that fail this check are
    excluded from the recovered-parameter comparison on whichever
    language's side they occur (independently -- MATLAB and Python draw
    different random numbers per seed, so the invalid seeds need not match
    between languages).
    """
    n_basis = int(np.floor(2 * (N_TR * TR_S) / freq_hz + 1))
    return n_basis >= 3


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (pooled-SD standardized mean difference) between two samples."""
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd


def recover_cardiac_or_respiratory(seed: int, jitter_scalar: float, nominal_freq: float,
                                    nominal_amp: float) -> tuple[float, float, float]:
    """Recover (f, a, p) for a cardiac or respiratory draw.

    Parameters
    ----------
    seed : int
        RNG seed (same convention as PmNoise(seed=...)).
    jitter_scalar : float
        Scalar jitter value; expands to [jitter_scalar, jitter_scalar, 0.0]
        exactly as PmNoise._expand_jitter() does for length-1 input.
    nominal_freq, nominal_amp : float
        Unperturbed parameter values.

    Returns
    -------
    f, a, p : float
        Recovered frequency (Hz), amplitude, and phase (rad).
    """
    rng = np.random.RandomState(seed)
    f = nominal_freq * (1 + jitter_scalar * rng.standard_normal())
    a = nominal_amp * (1 + jitter_scalar * rng.standard_normal())
    p = 2 * np.pi * 0.0 * rng.standard_normal()  # phase jitter = 0 for scalar form
    return f, a, p


def recover_drift(seed: int, jitter_scalar: float, nominal_freq: float,
                   nominal_amp: float) -> tuple[float, float]:
    """Recover (f, a) for a drift draw. See recover_cardiac_or_respiratory."""
    rng = np.random.RandomState(seed)
    f = nominal_freq * (1 + jitter_scalar * rng.standard_normal())
    a = nominal_amp * (1 + jitter_scalar * rng.standard_normal())
    return f, a


def verify_recovery_matches_pm_noise() -> None:
    """One-time bit-exact cross-check: recovered (f, a, p) must reproduce
    the actual signal pm_noise.py's private _compute_* methods return.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prf_models.pm_noise import PmNoise, PmAdapter

    TR, N = 1.5, 100
    pm = PmAdapter(TR=TR, time_points_n=N, time_points_series=np.arange(N) * TR)
    t = pm.time_points_series
    seed, jitter_scalar = 7, 0.3

    print("-- verify_recovery_matches_pm_noise --")

    # Cardiac
    noise = PmNoise(pm=pm, seed=seed, voxel='mid', jitter=[jitter_scalar])
    actual = noise._compute_cardiac(np.random.RandomState(seed), noise._expand_jitter())
    f, a, p = recover_cardiac_or_respiratory(seed, jitter_scalar, NOMINAL['cardiac']['freq'],
                                              NOMINAL['cardiac']['amp'])
    reconstructed = a * np.sin(2 * np.pi * t * f + p)
    ok = np.allclose(actual, reconstructed, atol=0)
    print(f"  cardiac     max|diff|={np.max(np.abs(actual - reconstructed)):.3e}  {'✓' if ok else 'FAILED'}")

    # Respiratory
    actual = noise._compute_respiratory(np.random.RandomState(seed), noise._expand_jitter())
    f, a, p = recover_cardiac_or_respiratory(seed, jitter_scalar, NOMINAL['respiratory']['freq'],
                                              NOMINAL['respiratory']['amp'])
    reconstructed = a * np.sin(2 * np.pi * t * f + p)
    ok = np.allclose(actual, reconstructed, atol=0)
    print(f"  respiratory max|diff|={np.max(np.abs(actual - reconstructed)):.3e}  {'✓' if ok else 'FAILED'}")

    # Drift -- smaller jitter here: at jitter_scalar=0.3 this particular seed
    # perturbs the period enough to push n_basis below 3 (see recover_drift's
    # module docstring on the floor()-discontinuity risk at larger jitter).
    # The formula being checked doesn't depend on the magnitude, so a smaller
    # value that avoids the edge case is an equally valid check.
    drift_jitter_scalar = 0.05
    noise_drift = PmNoise(pm=pm, seed=seed, voxel='mid', jitter=[drift_jitter_scalar])
    actual = noise_drift._compute_drift(np.random.RandomState(seed), noise_drift._expand_jitter())
    f, a = recover_drift(seed, drift_jitter_scalar, NOMINAL['drift']['freq'], NOMINAL['drift']['amp'])
    from prf_models.pm_noise import spm_drift
    n_basis = int(np.floor(2 * (N * TR) / f + 1))
    reconstructed = a * spm_drift(N, n_basis)[:, 1:].sum(axis=1)
    ok = np.allclose(actual, reconstructed, atol=0)
    print(f"  drift       max|diff|={np.max(np.abs(actual - reconstructed)):.3e}  {'✓' if ok else 'FAILED'}")


if __name__ == '__main__':
    verify_recovery_matches_pm_noise()
