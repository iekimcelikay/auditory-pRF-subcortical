"""
plot_noise_verification.py
Supervisor-ready verification of pm_noise.py against the MATLAB reference.

Run from project root:
    python tests/plot_noise_verification.py

Figures saved to:  tests/matlab_verification/
    fig1_deterministic.png  — Python vs MATLAB comparison (requires MATLAB outputs)
    fig2_spectra.png        — power spectra per voxel level (Python-only)
    fig3_snr.png            — measured SNR vs published targets (Python-only)

The MATLAB comparison (fig1) is only generated when matlab_verify_noise.m has
been run first and its CSV outputs are present.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── project path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from prf_models.pm_noise import PmNoise, PmAdapter, spm_drift, REFERENCE_BOLD_STD

MATLAB_DIR = Path(__file__).parent / 'matlab_verification'
MATLAB_DIR.mkdir(exist_ok=True)

HAVE_MATLAB = (MATLAB_DIR / 'spm_drift.csv').exists()
if not HAVE_MATLAB:
    warnings.warn(
        "\nMATLAB outputs not found — fig1_deterministic.png will be skipped.\n"
        "Run tests/matlab_verify_noise.m in MATLAB first, then re-run this script.\n"
    )

STYLE = dict(dpi=150, bbox_inches='tight')

# ─── shared parameters (must match matlab_verify_noise.m) ────────────────────
TR      = 1.5
N       = 100
T       = np.arange(N, dtype=float) * TR         # time axis (s)
N_BASIS = int(np.floor(2 * (N * TR) / 120 + 1))  # = 4

# 'mid' preset amplitudes / frequencies
A_CARDIAC  = 0.01;  F_CARDIAC  = 1.05   # Hz
A_RESP     = 0.01;  F_RESP     = 0.3    # Hz
A_DRIFT    = 0.01

pm_base = PmAdapter(TR=TR, time_points_n=N, time_points_series=T)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Deterministic comparison: Python vs MATLAB
# ═══════════════════════════════════════════════════════════════════════════════

if HAVE_MATLAB:
    # --- Python signals -------------------------------------------------------
    C_py       = spm_drift(N, N_BASIS)
    drift_py   = A_DRIFT * C_py[:, 1:].sum(axis=1)
    cardiac_py = A_CARDIAC * np.sin(2 * np.pi * T * F_CARDIAC)
    resp_py    = A_RESP    * np.sin(2 * np.pi * T * F_RESP)

    # --- MATLAB signals -------------------------------------------------------
    drift_ml   = np.loadtxt(MATLAB_DIR / 'drift_signal.csv',        delimiter=',')
    cardiac_ml = np.loadtxt(MATLAB_DIR / 'cardiac_jitter0.csv',     delimiter=',')
    resp_ml    = np.loadtxt(MATLAB_DIR / 'respiratory_jitter0.csv', delimiter=',')
    C_ml       = np.loadtxt(MATLAB_DIR / 'spm_drift.csv',           delimiter=',')

    components = [
        ('Drift signal',           drift_py,   drift_ml,   'steelblue'),
        ('Cardiac (jitter = 0)',   cardiac_py, cardiac_ml, 'darkorange'),
        ('Respiratory (jitter = 0)', resp_py,  resp_ml,    'seagreen'),
    ]

    fig1, axes1 = plt.subplots(3, 2, figsize=(13, 9))
    fig1.suptitle(
        'Deterministic components: Python port vs MATLAB reference\n'
        'jitter = [0, 0, 0] — no random draws, exact agreement expected',
        fontsize=12
    )

    for row, (label, py_sig, ml_sig, color) in enumerate(components):
        residual = py_sig - ml_sig
        max_err  = np.max(np.abs(residual))

        ax_overlay = axes1[row, 0]
        ax_resid   = axes1[row, 1]

        # Overlay
        ax_overlay.plot(T, ml_sig, color='firebrick', lw=2.5, ls='--',
                        label='MATLAB', zorder=2)
        ax_overlay.plot(T, py_sig, color=color, lw=1.5, alpha=0.85,
                        label='Python', zorder=3)
        ax_overlay.set_ylabel(label, fontsize=9)
        ax_overlay.legend(fontsize=8, loc='upper right')
        ax_overlay.grid(True, alpha=0.3)
        ax_overlay.set_title(
            f'{label}  —  max |Python − MATLAB| = {max_err:.2e}',
            fontsize=9
        )

        # Residual
        ax_resid.plot(T, residual, color='k', lw=1.0)
        ax_resid.axhline(0, color='grey', lw=0.8, ls='--')
        ax_resid.set_ylabel('Residual', fontsize=9)
        ax_resid.set_title(
            f'Residual  (expect ≈ 0 if exact match)', fontsize=9
        )
        ax_resid.grid(True, alpha=0.3)

    for ax in axes1[-1, :]:
        ax.set_xlabel('Time (s)')

    # Extra panel: spm_drift column norms
    py_norms = np.linalg.norm(C_py, axis=0)
    ml_norms = np.linalg.norm(C_ml, axis=0)
    fig1.text(
        0.5, 0.005,
        f'spm_drift column L2 norms — Python: {np.round(py_norms, 6)}  '
        f'MATLAB: {np.round(ml_norms, 6)}  (expected all = 1.0)',
        ha='center', fontsize=8, style='italic'
    )

    fig1.tight_layout(rect=[0, 0.02, 1, 1])
    out1 = MATLAB_DIR / 'fig1_deterministic.png'
    fig1.savefig(out1, **STYLE)
    print(f"Saved {out1}")
    plt.close(fig1)

    # --- Print table to console -----------------------------------------------
    print("\n── Deterministic comparison ──────────────────────────────────")
    print(f"{'Component':<28}  {'max |Py − ML|':>14}  {'mean |Py − ML|':>14}")
    print("-" * 60)
    for label, py_sig, ml_sig, _ in components:
        r = py_sig - ml_sig
        print(f"{label:<28}  {np.max(np.abs(r)):>14.3e}  {np.mean(np.abs(r)):>14.3e}")
    print()

else:
    print("Skipping fig1_deterministic.png (MATLAB outputs not found).\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Power spectra per voxel level
# For cardiac (1.05 Hz) to be visible, we use TR=0.4 s (Nyquist = 1.25 Hz).
# Nyquist for TR=1.5 is only 0.33 Hz — cardiac would alias.
# ═══════════════════════════════════════════════════════════════════════════════

TR_spec  = 0.4
N_spec   = 400        # 160 s total scan
T_spec   = np.arange(N_spec, dtype=float) * TR_spec
pm_spec  = PmAdapter(TR=TR_spec, time_points_n=N_spec, time_points_series=T_spec)
N_SEEDS  = 50

nyquist  = 0.5 / TR_spec   # 1.25 Hz
freqs    = np.fft.rfftfreq(N_spec, d=TR_spec)

levels   = [('low', '#2166ac'), ('mid', '#d6604d'), ('high', '#b2182b')]
fig2, axes2 = plt.subplots(3, 2, figsize=(13, 9))
fig2.suptitle(
    f'Stochastic noise properties per voxel level\n'
    f'TR = {TR_spec} s  |  {N_spec} TRs  |  Nyquist = {nyquist:.2f} Hz  '
    f'|  {N_SEEDS} realisations per level',
    fontsize=12
)

for row, (voxel, color) in enumerate(levels):
    ax_t = axes2[row, 0]
    ax_f = axes2[row, 1]

    all_spectra = []
    for seed in range(N_SEEDS):
        noise = PmNoise(pm=pm_spec, seed=seed, voxel=voxel)
        noise.compute()
        vals = noise.values

        if seed < 5:
            alpha = 0.8 if seed == 0 else 0.25
            lw    = 1.4 if seed == 0 else 0.7
            ax_t.plot(T_spec, vals, color=color, alpha=alpha, lw=lw)

        F = np.abs(np.fft.rfft(vals)) / N_spec
        F[1:-1] *= 2
        all_spectra.append(F)

    spectra   = np.array(all_spectra)
    mean_spec = spectra.mean(axis=0)
    std_spec  = spectra.std(axis=0)

    ax_f.fill_between(freqs, mean_spec - std_spec, mean_spec + std_spec,
                      color=color, alpha=0.25, label='±1 SD')
    ax_f.plot(freqs, mean_spec, color=color, lw=1.5, label='mean')

    # Mark expected spectral peaks
    peak_cardiac = PmNoise.defaults_get(voxel)['cardiac_frequency']
    peak_resp    = PmNoise.defaults_get(voxel)['respiratory_frequency']
    ymax = mean_spec.max() * 1.05 if mean_spec.max() > 0 else 1e-4

    for f_peak, f_lbl in [(peak_cardiac, 'cardiac'), (peak_resp, 'resp.')]:
        if f_peak <= nyquist:
            ax_f.axvline(f_peak, color='k', ls=':', lw=1.2, alpha=0.7)
            ax_f.text(f_peak, ymax * 0.92, f_lbl, fontsize=7, ha='center')

    ax_t.set_ylabel(f'{voxel}\namplitude', fontsize=9)
    ax_t.grid(True, alpha=0.3)
    ax_f.set_ylabel('amplitude', fontsize=9)
    ax_f.legend(fontsize=8, loc='upper right')
    ax_f.grid(True, alpha=0.3)

    if row == 0:
        ax_t.set_title('Time domain (first 5 seeds)', fontsize=10)
        ax_f.set_title('One-sided power spectrum', fontsize=10)

for ax in axes2[-1, :]:
    ax.set_xlabel('Time (s)' if ax == axes2[-1, 0] else 'Frequency (Hz)')

fig2.tight_layout()
out2 = MATLAB_DIR / 'fig2_spectra.png'
fig2.savefig(out2, **STYLE)
print(f"Saved {out2}")
plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — SNR calibration vs published targets
# Published targets (Lerma-Usabiaga et al. Fig 4):
#   low = +5.29 dB,  mid = −0.51 dB,  high = −4.29 dB
# ═══════════════════════════════════════════════════════════════════════════════

TARGETS = {'low': 5.29, 'mid': -0.51, 'high': -4.29}
N_SNR   = 300    # seeds per voxel level

pm_snr = PmAdapter(TR=TR, time_points_n=N, time_points_series=T)

print("── SNR calibration ───────────────────────────────────────────")
print(f"{'Voxel':<6}  {'mean (dB)':>10}  {'SD (dB)':>8}  {'target (dB)':>12}  {'diff':>6}")
print("-" * 48)

snr_data = {}
for voxel, _ in levels:
    snrs = []
    for seed in range(N_SNR):
        noise = PmNoise(pm=pm_snr, seed=seed, voxel=voxel)
        noise.compute()
        # SNR referenced to REFERENCE_BOLD_STD (= 3% PSC)
        snr_db = 20 * np.log10(REFERENCE_BOLD_STD / noise.values.std())
        snrs.append(snr_db)
    snr_data[voxel] = np.array(snrs)
    m, s = snr_data[voxel].mean(), snr_data[voxel].std()
    diff  = m - TARGETS[voxel]
    print(f"{voxel:<6}  {m:>10.2f}  {s:>8.2f}  {TARGETS[voxel]:>12.2f}  {diff:>+6.2f}")

print()

fig3, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(
    f'SNR calibration — Python port vs published targets (Lerma-Usabiaga et al.)\n'
    f'{N_SNR} seeds per voxel level  |  reference BOLD std = {REFERENCE_BOLD_STD:.1%} PSC',
    fontsize=11
)

voxel_names = ['low', 'mid', 'high']
colors_fig  = [c for _, c in levels]
x           = np.arange(3)

for i, (voxel, color) in enumerate(levels):
    snrs = snr_data[voxel]
    ax_bar.bar(i, snrs.mean(), width=0.45, color=color, alpha=0.75,
               yerr=snrs.std(), capsize=6, label=voxel, ecolor='black')
    ax_bar.plot(i, TARGETS[voxel], 'k*', ms=14, zorder=5,
                label='_target' if i > 0 else 'published target')

    # Distribution
    ax_dist.hist(snrs, bins=30, color=color, alpha=0.5, label=voxel, density=True)
    ax_dist.axvline(TARGETS[voxel], color=color, ls='--', lw=1.5)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(voxel_names)
ax_bar.set_ylabel('SNR (dB)')
ax_bar.set_title('Mean ± SD per voxel level  (★ = published target)')
ax_bar.axhline(0, color='k', lw=0.8, alpha=0.4)
ax_bar.legend(fontsize=9)
ax_bar.grid(True, axis='y', alpha=0.3)

ax_dist.set_xlabel('SNR (dB)')
ax_dist.set_ylabel('density')
ax_dist.set_title('SNR distribution per voxel level  (dashed = target)')
ax_dist.legend(fontsize=9)
ax_dist.grid(True, alpha=0.3)

fig3.tight_layout()
out3 = MATLAB_DIR / 'fig3_snr.png'
fig3.savefig(out3, **STYLE)
print(f"Saved {out3}")
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nAll outputs in: {MATLAB_DIR.resolve()}")
if not HAVE_MATLAB:
    print("\nTo generate fig1_deterministic.png:")
    print("  1. In MATLAB: run('tests/matlab_verify_noise.m')")
    print("  2. Re-run this script")
