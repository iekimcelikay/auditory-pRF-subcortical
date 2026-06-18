from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from find_optimal_durations import find_closest_durations


# ── ERB scale (Moore & Glasberg 1990) ────────────────────────────────────────

def hz_to_erb(f: float | np.ndarray) -> np.ndarray:
    """Convert Hz to ERB-rate (Cams)."""
    return 21.4 * np.log10(1.0 + np.asarray(f) * 0.00437)

def erb_to_hz(e: float | np.ndarray) -> np.ndarray:
    """Convert ERB-rate (Cams) to Hz."""
    return (np.power(10.0, np.asarray(e) / 21.4) - 1.0) / 0.00437


# ── Filter bank parameters ────────────────────────────────────────────────────

@dataclass
class GaussianFilterBank:
    means_hz:   np.ndarray   # centre frequencies in Hz,  shape (n,)
    means_erb:  np.ndarray   # centre frequencies in Cam, shape (n,)
    sigma:      float        # shared σ in native space (Hz or Cam)
    sigma_unit: str          # 'Hz' or 'Cam'
    bounds_hz:  np.ndarray   # cutoff frequencies in Hz,  shape (n+1,)
    bounds_erb: np.ndarray   # cutoff frequencies in Cam, shape (n+1,)
    n:          int          # number of Gaussians
    k:          float        # σ-factor: boundaries at ±k·σ from each mean
    mode:       str          # 'erb' or 'linear'


def fit_gaussian_filterbank(
    min_freq: float = 450.0,
    max_freq: float = 1600.0,
    n:        int   = 3,
    k:        float = 2.0,
    mode:     str   = 'erb',
) -> GaussianFilterBank:
    """
    Fit n equally spaced, equal-width Gaussians spanning [min_freq, max_freq].

    The range is divided into n equal segments in the chosen spacing domain.
    Each Gaussian is centred at the midpoint of its segment. σ is chosen so
    that each boundary sits exactly at ±k·σ from the adjacent mean — i.e.
    the dashed cutoff lines in the explorer.

    Parameters
    ----------
    min_freq, max_freq : float
        Frequency range in Hz. Defaults: 450, 1600.
    n : int
        Number of Gaussians.
    k : float
        σ-factor. At a boundary, response = exp(−k²/2) of peak.
        k=2 → ~13.5% of peak at cutoff.
    mode : str
        'erb'    — spacing and σ in ERB (Cam) space (recommended).
        'linear' — spacing and σ in Hz.

    Returns
    -------
    GaussianFilterBank
    """
    if mode == 'erb':
        mn_e = hz_to_erb(min_freq)
        mx_e = hz_to_erb(max_freq)
        w    = (mx_e - mn_e) / n
        sigma = float(w / (2.0 * k))

        mu_erbs    = np.array([mn_e + w * (i + 0.5) for i in range(n)])
        bound_erbs = np.array([mn_e + w * i          for i in range(n + 1)])
        mu_hz      = erb_to_hz(mu_erbs)
        bound_hz   = np.asarray(erb_to_hz(bound_erbs))
        bound_hz[0]  = min_freq   # pin exact edges (avoid float drift)
        bound_hz[-1] = max_freq

        return GaussianFilterBank(
            means_hz=mu_hz, means_erb=mu_erbs,
            sigma=sigma, sigma_unit='Cam',
            bounds_hz=bound_hz, bounds_erb=bound_erbs,
            n=n, k=k, mode='erb',
        )

    else:  # linear
        w     = (max_freq - min_freq) / n
        sigma = w / (2.0 * k)

        mu_hz    = np.array([min_freq + w * (i + 0.5) for i in range(n)])
        bound_hz = np.array([min_freq + w * i          for i in range(n + 1)])

        return GaussianFilterBank(
            means_hz=mu_hz, means_erb=hz_to_erb(mu_hz),
            sigma=sigma, sigma_unit='Hz',
            bounds_hz=bound_hz, bounds_erb=hz_to_erb(bound_hz),
            n=n, k=k, mode='linear',
        )


# ── Timing ────────────────────────────────────────────────────────────────────

def calculate_num_tones(
    tone_duration:  float,
    isi:            float,
    total_duration: float,
    sample_rate:    int,
) -> tuple[int, int, int]:
    """
    Standalone version of the timing helper.

    Parameters
    ----------
    tone_duration, isi, total_duration : float
        All in seconds.
    sample_rate : int
        In Hz.

    Returns
    -------
    num_tones : int
    isi_samples : int
    total_samples : int
    """
    total_samples = int(total_duration * sample_rate)
    isi_samples   = int(isi            * sample_rate)
    tone_samples  = int(tone_duration  * sample_rate)
    num_tones     = int(total_samples // (tone_samples + isi_samples))
    return num_tones, isi_samples, total_samples


# ── Sampling ──────────────────────────────────────────────────────────────────

def _bell_curve_counts(
    bin_centers: np.ndarray,
    mu:          float,
    sigma:       float,
    n_total:     int,
) -> np.ndarray:
    """
    Allocate n_total samples across bins with a symmetric monotone Gaussian shape.

    Weights come from the Gaussian PDF at each bin centre, forced symmetric,
    then converted to integers via largest-remainder rounding. A monotone clamp
    ensures no bin exceeds its inner neighbour.

    Returns
    -------
    counts : np.ndarray, shape (n_bins,), dtype int, sums to n_total
    """
    n = len(bin_centers)
    w = np.exp(-(bin_centers - mu) ** 2 / (2.0 * sigma ** 2))
    for i in range(n // 2):
        avg = (w[i] + w[n - 1 - i]) / 2.0
        w[i] = w[n - 1 - i] = avg
    w /= w.sum()

    exact  = w * n_total
    counts = np.floor(exact).astype(int)
    n_extra = n_total - counts.sum()
    if n_extra > 0:
        counts[np.argsort(exact - counts)[-n_extra:]] += 1

    peak = int(np.argmax(counts))
    for i in range(peak - 1, -1, -1):
        counts[i] = min(counts[i], counts[i + 1])
    for i in range(peak + 1, n):
        counts[i] = min(counts[i], counts[i - 1])
    counts[peak] += n_total - counts.sum()

    return counts


def sample_frequencies(
    filterbank:     GaussianFilterBank,
    gaussian_idx:   int,
    tone_duration:  float,
    isi:            float,
    total_duration: float,
    sample_rate:    int,
    round_to_hz:    bool = False,
    unique:         bool = False,
    stratified:     bool = True,
    rng:            np.random.Generator | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Sample tone frequencies from the i-th truncated Gaussian.

    Calls calculate_num_tones internally to determine how many tones fit
    in the trial, then draws exactly that many samples.

    Parameters
    ----------
    filterbank : GaussianFilterBank
    gaussian_idx : int
        Which Gaussian to sample from (0-indexed).
    tone_duration, isi, total_duration : float
        All in seconds. Passed directly to calculate_num_tones.
    sample_rate : int
        In Hz. Passed directly to calculate_num_tones.
    round_to_hz : bool
        Round output to nearest integer Hz. Required for unique=True to be
        meaningful; has no effect at floating-point precision.
    unique : bool
        Reject duplicate values (after rounding, if round_to_hz=True).
        Uses a rejection-sampling loop; safe as long as num_tones is well
        below the number of distinct values in the support.
    stratified : bool
        If True (default), divide the distribution into n equal probability
        bins and draw one sample per bin. Prevents clustering when num_tones
        is small. If False, draw independently (pure random).
    rng : np.random.Generator, optional
        Pass a seeded generator for reproducibility, e.g. np.random.default_rng(42).

    Returns
    -------
    freqs_hz : np.ndarray, shape (num_tones,)
        Sampled tone frequencies in Hz.
    num_tones : int
        Number of tones that fit in the trial (from calculate_num_tones).
    isi_samples : int
        ISI expressed in samples (from calculate_num_tones).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_tones, isi_samples, _ = calculate_num_tones(
        tone_duration, isi, total_duration, sample_rate,
    )

    i = gaussian_idx
    if filterbank.mode == 'erb':
        mu    = filterbank.means_erb[i]
        sigma = filterbank.sigma
        lo    = filterbank.bounds_erb[i]
        hi    = filterbank.bounds_erb[i + 1]
    else:
        mu    = filterbank.means_hz[i]
        sigma = filterbank.sigma
        lo    = filterbank.bounds_hz[i]
        hi    = filterbank.bounds_hz[i + 1]

    a = (lo - mu) / sigma   # = −k by construction
    b = (hi - mu) / sigma   # = +k by construction

    def draw(n: int) -> np.ndarray:
        if stratified:
            native_edges = np.linspace(lo, hi, n + 1)
            bin_centers  = (native_edges[:-1] + native_edges[1:]) / 2
            counts = _bell_curve_counts(bin_centers, mu, sigma, n)
            raw = np.concatenate([
                rng.uniform(native_edges[i], native_edges[i + 1], size=int(counts[i]))
                for i in range(n)
            ])
            rng.shuffle(raw)
        else:
            raw = np.asarray(stats.truncnorm.rvs(
                a, b, loc=mu, scale=sigma, size=n, random_state=rng,
            ))
        out = erb_to_hz(raw) if filterbank.mode == 'erb' else raw
        return np.round(out).astype(int) if round_to_hz else out

    if not unique:
        return draw(num_tones), num_tones, isi_samples

    # Rejection loop — oversample until num_tones unique values collected
    seen: set = set()
    collected: list = []
    while len(collected) < num_tones:
        for v in draw(num_tones * 2):
            key = int(v) if round_to_hz else v
            if key not in seen:
                seen.add(key)
                collected.append(v)
            if len(collected) == num_tones:
                break

    return np.array(collected), num_tones, isi_samples

# ── Master frequency list ─────────────────────────────────────────────────────

@dataclass
class MasterFrequencyList:
    freqs_by_bin: list[np.ndarray]  # sorted native-space freqs per bin
    bin_centers:  np.ndarray        # bin centres in native space
    native_edges: np.ndarray        # bin edges in native space
    gaussian_idx: int
    filterbank:   GaussianFilterBank


def build_master_list(
    filterbank:   GaussianFilterBank,
    gaussian_idx: int,
    n_bins:       int = 10,
    n_master:     int = 1000,
    seed:         int = 0,
) -> MasterFrequencyList:
    """
    Build a fixed frequency pool for one Gaussian with bell-curve counts.

    Parameters
    ----------
    filterbank : GaussianFilterBank
    gaussian_idx : int
        Which Gaussian to build the pool for (0-indexed).
    n_bins : int
        Number of equal-width ERB bins across the Gaussian's support.
    n_master : int
        Total frequencies in the pool. Must be >= the largest n_tones
        that will ever be requested via sample_from_master.
    seed : int
        RNG seed. Same seed always produces the same master list.
    """
    rng = np.random.default_rng(seed)
    g   = gaussian_idx

    if filterbank.mode == 'erb':
        lo, hi = filterbank.bounds_erb[g], filterbank.bounds_erb[g + 1]
        mu     = float(filterbank.means_erb[g])
    else:
        lo, hi = filterbank.bounds_hz[g], filterbank.bounds_hz[g + 1]
        mu     = float(filterbank.means_hz[g])

    native_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers  = (native_edges[:-1] + native_edges[1:]) / 2
    counts       = _bell_curve_counts(bin_centers, mu, filterbank.sigma, n_master)

    freqs_by_bin = [
        np.sort(rng.uniform(native_edges[i], native_edges[i + 1], size=int(counts[i])))
        for i in range(n_bins)
    ]

    return MasterFrequencyList(
        freqs_by_bin=freqs_by_bin,
        bin_centers=bin_centers,
        native_edges=native_edges,
        gaussian_idx=gaussian_idx,
        filterbank=filterbank,
    )


def sample_from_master(
    master:         MasterFrequencyList,
    n_tones:        int,
    condition_seed: int,
) -> np.ndarray:
    """
    Deterministically draw n_tones from the master list with bell-curve shape.

    The same (master, n_tones, condition_seed) triple always returns identical
    frequencies in identical order.

    Parameters
    ----------
    master : MasterFrequencyList
    n_tones : int
        Number of frequencies to select. Must be <= master's n_master.
    condition_seed : int
        Seed controlling the shuffle order. Derive from condition parameters
        so each condition gets a unique but reproducible ordering, e.g.
        gaussian_idx * 100_000 + int(round(dur_ms * 100)).
    """
    fb = master.filterbank
    g  = master.gaussian_idx
    mu = float(fb.means_erb[g] if fb.mode == 'erb' else fb.means_hz[g])

    counts = _bell_curve_counts(master.bin_centers, mu, fb.sigma, n_tones)

    freqs_native = np.concatenate([
        master.freqs_by_bin[i][:int(min(counts[i], len(master.freqs_by_bin[i])))]
        for i in range(len(master.freqs_by_bin))
    ])

    np.random.default_rng(condition_seed).shuffle(freqs_native)
    return erb_to_hz(freqs_native) if fb.mode == 'erb' else freqs_native


# ── Stimulus timing ───────────────────────────────────────────────────────────

TOTAL_DURATION_S  = 20.0
_TARGET_DURS_MS   = (35, 45, 60, 75, 100, 150, 250, 500)
ISI_MS_SINGLE     = 75
TONE_ON_MS        = find_closest_durations(
    _TARGET_DURS_MS,
    isi_ms=ISI_MS_SINGLE,
    seq_dur_s=TOTAL_DURATION_S,
    dur_max_ms=max(_TARGET_DURS_MS),
)
ISI_MS            = (ISI_MS_SINGLE,) * len(TONE_ON_MS)
SAMPLE_RATE       = 100_000            # Hz


def sample_all_conditions(
    filterbank:  GaussianFilterBank,
    tone_on_ms:  tuple[float, ...] = TONE_ON_MS,
    isi_ms:      tuple[float, ...] = ISI_MS,
    total_dur_s: float             = TOTAL_DURATION_S,
    sample_rate: int               = SAMPLE_RATE,
    round_to_hz: bool              = False,
    unique:      bool              = False,
    stratified:  bool              = True,
    masters:     list[MasterFrequencyList] | None = None,
    rng:         np.random.Generator | None = None,
) -> dict[tuple[int, float], np.ndarray]:
    """
    Sample tone frequencies for every (gaussian, duration) combination.

    Parameters
    ----------
    filterbank : GaussianFilterBank
    tone_on_ms, isi_ms : tuple of float
        Paired duration and ISI values in ms, matched index-by-index.
    total_dur_s : float
        Sequence duration in seconds.
    sample_rate : int
        In Hz.
    round_to_hz, unique : bool
        Passed to sample_frequencies (ignored when masters is provided).
    masters : list of MasterFrequencyList, optional
        One per Gaussian (len == filterbank.n). When provided, each condition
        always draws the same frequencies deterministically from the master pool.
        When None, falls back to stochastic sample_frequencies.
    rng : np.random.Generator, optional
        Used only when masters is None.

    Returns
    -------
    dict mapping (gaussian_idx, dur_ms) -> freqs_hz array
    """
    if masters is None and rng is None:
        rng = np.random.default_rng()

    results: dict[tuple[int, float], np.ndarray] = {}
    for dur_ms, isi_ms_val in zip(tone_on_ms, isi_ms):
        for g_idx in range(filterbank.n):
            if masters is not None:
                n_tones, _, _ = calculate_num_tones(
                    dur_ms / 1000.0, isi_ms_val / 1000.0, total_dur_s, sample_rate,
                )
                cond_seed = g_idx * 100_000 + int(round(dur_ms * 100))
                freqs = sample_from_master(masters[g_idx], n_tones, cond_seed)
            else:
                freqs, _, _ = sample_frequencies(
                    filterbank=filterbank,
                    gaussian_idx=g_idx,
                    tone_duration=dur_ms / 1000.0,
                    isi=isi_ms_val / 1000.0,
                    total_duration=total_dur_s,
                    sample_rate=sample_rate,
                    round_to_hz=round_to_hz,
                    unique=unique,
                    stratified=stratified,
                    rng=rng,
                )
            results[(g_idx, dur_ms)] = freqs
    return results


# ── Inspection ────────────────────────────────────────────────────────────────

def filterbank_to_dataframe(filterbank: GaussianFilterBank) -> pd.DataFrame:
    """Return a per-channel summary of the filter bank as a DataFrame."""
    return pd.DataFrame({
        'means_hz':  filterbank.means_hz,
        'means_erb': filterbank.means_erb,
        'bounds_lo_hz': filterbank.bounds_hz[:-1],
        'bounds_hi_hz': filterbank.bounds_hz[1:],
        'bounds_lo_erb': filterbank.bounds_erb[:-1],
        'bounds_hi_erb': filterbank.bounds_erb[1:],
    })


def plot_frequency_histograms(
    freqs_hz:     np.ndarray,
    filterbank:   GaussianFilterBank,
    gaussian_idx: int,
    n_bins:       int = 30,
) -> None:
    """
    Plot the sampled frequency distribution in both ERB and Hz space.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Sampled frequencies in Hz (e.g. from sample_from_master).
    filterbank : GaussianFilterBank
    gaussian_idx : int
        Which Gaussian the frequencies were drawn from.
    n_bins : int
        Number of histogram bins in each panel.
    """
    import matplotlib.pyplot as plt

    g        = gaussian_idx
    lo_hz    = float(filterbank.bounds_hz[g])
    hi_hz    = float(filterbank.bounds_hz[g + 1])
    lo_erb   = float(filterbank.bounds_erb[g])
    hi_erb   = float(filterbank.bounds_erb[g + 1])
    freqs_erb = hz_to_erb(freqs_hz)

    erb_edges = np.linspace(lo_erb, hi_erb, n_bins + 1)
    hz_edges  = np.linspace(lo_hz,  hi_hz,  n_bins + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    ax1.hist(freqs_erb, bins=erb_edges, color='steelblue', edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('ERB-rate (Cam)')
    ax1.set_ylabel('Count')
    ax1.set_title('ERB space  (equal-width bins)')

    ax2.hist(freqs_hz, bins=hz_edges, color='coral', edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Hz space  (equal-width bins)')

    fig.suptitle(
        f'Gaussian {gaussian_idx}   n = {len(freqs_hz)} tones'
        f'   [{lo_hz:.0f}–{hi_hz:.0f} Hz]'
    )
    plt.tight_layout()
    plt.show()