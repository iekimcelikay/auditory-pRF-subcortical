"""
plot_gaussian_filterbank.py

Plots the Gaussian filterbank that determines tone-cloud centre frequencies,
in both Hz and ERB (Cam) space, side by side.

This is a standalone re-implementation of `fit_gaussian_filterbank` from
sample_tone_cloud_freqs_gaussian.py (copied verbatim, not reinvented) so
this script has no dependency on find_optimal_durations.py, which is not
needed for this figure. If your actual filterbank config differs from the
FREQ_MIN_HZ / FREQ_MAX_HZ / N_GAUSSIANS / K_SIGMA values below, edit the
CONFIG block — do not hand-edit the curves.

Usage
-----
    python plot_gaussian_filterbank.py
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG — must match save_tone_clouds_gaussian_prf.py
# ==============================================================================

FREQ_MIN_HZ = 450.0
FREQ_MAX_HZ = 1600.0
N_GAUSSIANS = 3
K_SIGMA     = 2.0      # boundaries at +/- k*sigma from each mean
MODE        = 'erb'    # 'erb' (actual sampling space) or 'linear'

OUT_HZ_PNG  = "filterbank_hz.png"
OUT_ERB_PNG = "filterbank_erb.png"
OUT_COMBINED_PNG = "filterbank_combined.png"


# ==============================================================================
# ERB scale (Moore & Glasberg 1990) — copied from sample_tone_cloud_freqs_gaussian.py
# ==============================================================================

def hz_to_erb(f):
    return 21.4 * np.log10(1.0 + np.asarray(f) * 0.00437)

def erb_to_hz(e):
    return (np.power(10.0, np.asarray(e) / 21.4) - 1.0) / 0.00437


@dataclass
class GaussianFilterBank:
    means_hz:   np.ndarray
    means_erb:  np.ndarray
    sigma:      float
    sigma_unit: str
    bounds_hz:  np.ndarray
    bounds_erb: np.ndarray
    n:          int
    k:          float
    mode:       str


def fit_gaussian_filterbank(min_freq=450.0, max_freq=1600.0, n=3, k=2.0, mode='erb'):
    """Verbatim copy of the function in sample_tone_cloud_freqs_gaussian.py."""
    if mode == 'erb':
        mn_e = hz_to_erb(min_freq)
        mx_e = hz_to_erb(max_freq)
        w    = (mx_e - mn_e) / n
        sigma = float(w / (2.0 * k))

        mu_erbs    = np.array([mn_e + w * (i + 0.5) for i in range(n)])
        bound_erbs = np.array([mn_e + w * i          for i in range(n + 1)])
        mu_hz      = erb_to_hz(mu_erbs)
        bound_hz   = np.asarray(erb_to_hz(bound_erbs))
        bound_hz[0]  = min_freq
        bound_hz[-1] = max_freq

        return GaussianFilterBank(
            means_hz=mu_hz, means_erb=mu_erbs,
            sigma=sigma, sigma_unit='Cam',
            bounds_hz=bound_hz, bounds_erb=bound_erbs,
            n=n, k=k, mode='erb',
        )
    else:
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


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_filterbank(fb: GaussianFilterBank, save_combined=True):
    """
    Plot the filterbank as normalized Gaussian curves (peak = 1), one panel
    in Hz, one in ERB (Cam) space. Dashed vertical lines mark the +/-k*sigma
    truncation boundaries; shaded regions mark each band's support.
    """
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, fb.n))

    fig, (ax_erb, ax_hz) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- ERB panel (native sampling space: symmetric bells) ----
    e_grid = np.linspace(fb.bounds_erb[0] - 0.5, fb.bounds_erb[-1] + 0.5, 2000)
    for i in range(fb.n):
        curve = gaussian_pdf(e_grid, fb.means_erb[i], fb.sigma)
        ax_erb.plot(e_grid, curve, color=colors[i], lw=2,
                    label=f"Band {i+1} ({fb.means_hz[i]:.0f} Hz)")
        ax_erb.fill_between(e_grid, curve, where=(e_grid >= fb.bounds_erb[i]) &
                             (e_grid <= fb.bounds_erb[i+1]),
                             color=colors[i], alpha=0.15)
        ax_erb.axvline(fb.means_erb[i], color=colors[i], ls=':', lw=1, alpha=0.7)
    for b in fb.bounds_erb:
        ax_erb.axvline(b, color='0.4', ls='--', lw=0.8)
    ax_erb.set_xlabel("ERB-rate (Cam)")
    ax_erb.set_ylabel("Normalized density")
    ax_erb.set_title(f"ERB space (sampling domain)\n$\\sigma$ = {fb.sigma:.3f} Cam, k = {fb.k}")
    ax_erb.set_ylim(0, 1.15)

    # ---- Hz panel (same curves, x-axis transformed — will look skewed) ----
    hz_grid = np.linspace(fb.bounds_hz[0] * 0.9, fb.bounds_hz[-1] * 1.05, 4000)
    erb_of_hz_grid = hz_to_erb(hz_grid)
    for i in range(fb.n):
        curve = gaussian_pdf(erb_of_hz_grid, fb.means_erb[i], fb.sigma)
        ax_hz.plot(hz_grid, curve, color=colors[i], lw=2,
                   label=f"Band {i+1} ({fb.means_hz[i]:.0f} Hz)")
        ax_hz.fill_between(hz_grid, curve, where=(hz_grid >= fb.bounds_hz[i]) &
                            (hz_grid <= fb.bounds_hz[i+1]),
                            color=colors[i], alpha=0.15)
        ax_hz.axvline(fb.means_hz[i], color=colors[i], ls=':', lw=1, alpha=0.7)
    for b in fb.bounds_hz:
        ax_hz.axvline(b, color='0.4', ls='--', lw=0.8)
    ax_hz.set_xlabel("Frequency (Hz)")
    ax_hz.set_title("Hz space (equal-width in ERB, so unequal & skewed in Hz)")
    ax_hz.set_ylim(0, 1.15)
    ax_hz.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Gaussian filterbank for tone-cloud centre frequencies "
        f"({fb.n} bands, {FREQ_MIN_HZ:.0f}\u2013{FREQ_MAX_HZ:.0f} Hz, "
        f"boundaries at \u00b1{fb.k}\u03c3)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_combined:
        fig.savefig(OUT_COMBINED_PNG, dpi=300, bbox_inches='tight')
        print(f"Saved: {OUT_COMBINED_PNG}")

    # Also save each panel individually, in case the report wants them separate
    for ax, fname, title in [
        (ax_erb, OUT_ERB_PNG, "ERB space"),
        (ax_hz, OUT_HZ_PNG, "Hz space"),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(6.5, 4.5))
        if fname == OUT_ERB_PNG:
            for i in range(fb.n):
                curve = gaussian_pdf(e_grid, fb.means_erb[i], fb.sigma)
                ax_single.plot(e_grid, curve, color=colors[i], lw=2,
                                label=f"Band {i+1} ({fb.means_hz[i]:.0f} Hz)")
                ax_single.fill_between(e_grid, curve,
                                        where=(e_grid >= fb.bounds_erb[i]) & (e_grid <= fb.bounds_erb[i+1]),
                                        color=colors[i], alpha=0.15)
                ax_single.axvline(fb.means_erb[i], color=colors[i], ls=':', lw=1, alpha=0.7)
            for b in fb.bounds_erb:
                ax_single.axvline(b, color='0.4', ls='--', lw=0.8)
            ax_single.set_xlabel("ERB-rate (Cam)")
        else:
            for i in range(fb.n):
                curve = gaussian_pdf(erb_of_hz_grid, fb.means_erb[i], fb.sigma)
                ax_single.plot(hz_grid, curve, color=colors[i], lw=2,
                                label=f"Band {i+1} ({fb.means_hz[i]:.0f} Hz)")
                ax_single.fill_between(hz_grid, curve,
                                        where=(hz_grid >= fb.bounds_hz[i]) & (hz_grid <= fb.bounds_hz[i+1]),
                                        color=colors[i], alpha=0.15)
                ax_single.axvline(fb.means_hz[i], color=colors[i], ls=':', lw=1, alpha=0.7)
            for b in fb.bounds_hz:
                ax_single.axvline(b, color='0.4', ls='--', lw=0.8)
            ax_single.set_xlabel("Frequency (Hz)")
        ax_single.set_ylabel("Normalized density")
        ax_single.set_ylim(0, 1.15)
        ax_single.set_title(f"{title}: tone-cloud filterbank ({fb.n} bands)")
        ax_single.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig_single.tight_layout()
        fig_single.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved: {fname}")

    return fig


def main():
    fb = fit_gaussian_filterbank(FREQ_MIN_HZ, FREQ_MAX_HZ, N_GAUSSIANS, K_SIGMA, MODE)

    print("Filterbank summary:")
    print(f"  mode  = {fb.mode}")
    print(f"  n     = {fb.n} bands")
    print(f"  sigma = {fb.sigma:.4f} {fb.sigma_unit}  (k = {fb.k})")
    for i in range(fb.n):
        print(f"  Band {i+1}: mean = {fb.means_hz[i]:7.1f} Hz "
              f"({fb.means_erb[i]:.3f} Cam)   "
              f"bounds = [{fb.bounds_hz[i]:.1f}, {fb.bounds_hz[i+1]:.1f}] Hz")

    plot_filterbank(fb)
    plt.show()


if __name__ == "__main__":
    main()
