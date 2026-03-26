"""
Recreates Fig. 1 from the AdapTrans paper using your AN pipeline.
Panels:
  A  – ON/OFF kernel stem plot (K=10, w=0.5, a=0.6)
  B  – Step-response time traces + three heatmaps (input, ON, OFF)
  C  – Bode magnitude plots for ON/OFF filters
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import freqz

# ── local import ────────────────────────────────────────────────────────────
from adaptrans_onoff_filters import (
    build_ON_kernel, build_OFF_kernel,
    apply_adaptrans, tau_to_a, willmore_tau,
)

# ════════════════════════════════════════════════════════════════════════════
# Shared parameters
# ════════════════════════════════════════════════════════════════════════════
W_PAPER   = 0.5          # adaptation weight used in paper figure
A_PAPER   = 0.6          # decay rate used in paper figure
K_PAPER   = 10           # kernel length used in paper figure

# ════════════════════════════════════════════════════════════════════════════
# Panel A – kernel stem plot
# ════════════════════════════════════════════════════════════════════════════
def panel_A(ax):
    kernel_ON  = build_ON_kernel( A_PAPER, W_PAPER, K_PAPER)
    kernel_OFF = build_OFF_kernel(A_PAPER, W_PAPER, K_PAPER)

    x = np.arange(K_PAPER)

    # stems for ON (red) and OFF (blue)
    markerline, stemlines, baseline = ax.stem(
        x, kernel_ON, linefmt='r-', markerfmt='ro', basefmt='k-', label='ON')
    plt.setp(stemlines, linewidth=1.2)
    plt.setp(markerline, markersize=5)

    markerline2, stemlines2, _ = ax.stem(
        x, kernel_OFF, linefmt='b-', markerfmt='bo', basefmt='k-', label='OFF')
    plt.setp(stemlines2, linewidth=1.2)
    plt.setp(markerline2, markersize=5)

    # annotations
    ax.axhline(0, color='k', linewidth=0.8)
    ax.annotate('', xy=(K_PAPER - 1, kernel_ON[-1]),
                xytext=(K_PAPER - 1, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(K_PAPER - 1.2, kernel_ON[-1] / 2, '1', color='red',
            ha='right', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(K_PAPER - 1, kernel_OFF[-1]),
                xytext=(K_PAPER - 1, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(K_PAPER - 1.2, kernel_OFF[-1] / 2, '-w', color='blue',
            ha='right', va='center', fontsize=9)

    # label the -w trough region for ON kernel
    trough_idx = 0
    ax.annotate('', xy=(trough_idx, kernel_ON[0]),
                xytext=(trough_idx + 4, kernel_ON[0]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
    ax.text(2, kernel_ON[0] * 1.15, '-w', color='red',
            ha='center', va='top', fontsize=9)

    ax.set_xlabel('(past)' + ' ' * 25 + '(present)', fontsize=8)
    ax.set_ylabel('Amplitude')
    ax.set_xlim(-0.5, K_PAPER + 0.5)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax.set_title('A.', loc='left', fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
# Panel B – step-response traces + heatmaps
# ════════════════════════════════════════════════════════════════════════════
def make_step_signal(N_CFs=128, T=150, onset=20, offset=80):
    """Rectangular block (step on / step off) for all CFs, amplitude=1."""
    sig = np.zeros((N_CFs, T))
    sig[:, onset:offset] = 1.0
    return sig

def panel_B_traces(ax, on_out, off_out, input_sig, onset=20, offset=80):
    T = input_sig.shape[1]
    t = np.arange(T)
    # pick the middle CF for the trace
    cf_idx = input_sig.shape[0] // 2

    ax.plot(t, input_sig[cf_idx], 'k-',  label='input', linewidth=1.5)
    ax.plot(t, on_out[cf_idx],   'r-',  label='ON',    linewidth=1.5)
    ax.plot(t, off_out[cf_idx],  'b-',  label='OFF',   linewidth=1.5)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Amplitude [a.u.]')
    ax.set_ylim(-0.7, 1.15)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_title('B.', loc='left', fontweight='bold')

def panel_B_heatmap(ax, data, title='', cmap='RdYlBu_r', diverging=False):
    if diverging:
        vmax = np.max(np.abs(data))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(data, aspect='auto', origin='lower',
                       cmap=cmap, norm=norm, interpolation='nearest')
    else:
        im = ax.imshow(data, aspect='auto', origin='lower',
                       cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('Frequency [bins]')
    return im


# ════════════════════════════════════════════════════════════════════════════
# Panel C – Bode plots
# ════════════════════════════════════════════════════════════════════════════
def panel_C(ax_on, ax_off):
    # vary w
    w_vals   = [0.1, 0.6, 1.0]
    w_styles = ['-', '--', ':']
    # vary tau (=> vary a at fixed dt=1 sample)
    tau_vals  = [5, 20]       # in samples (dt=1)
    tau_colors = ['green', 'red']

    a_fixed = 0.6  # for w-variation curves

    for w, ls in zip(w_vals, w_styles):
        k_on  = build_ON_kernel( a_fixed, w, K=200)
        k_off = build_OFF_kernel(a_fixed, w, K=200)
        w_arr, h_on  = freqz(k_on,  worN=512)
        _,     h_off = freqz(k_off, worN=512)
        mag_on  = 20 * np.log10(np.maximum(np.abs(h_on),  1e-12))
        mag_off = 20 * np.log10(np.maximum(np.abs(h_off), 1e-12))
        ax_on.plot(w_arr, mag_on,  color='k', linestyle=ls,
                   label=f'w={w}', linewidth=1.2)
        ax_off.plot(w_arr, mag_off, color='k', linestyle=ls,
                    label=f'w={w}', linewidth=1.2)

    w_fixed = 0.6  # for tau-variation curves
    for tau, color in zip(tau_vals, tau_colors):
        a_t   = np.exp(-1.0 / tau)          # dt=1 sample
        k_on  = build_ON_kernel( a_t, w_fixed, K=200)
        k_off = build_OFF_kernel(a_t, w_fixed, K=200)
        w_arr, h_on  = freqz(k_on,  worN=512)
        _,     h_off = freqz(k_off, worN=512)
        mag_on  = 20 * np.log10(np.maximum(np.abs(h_on),  1e-12))
        mag_off = 20 * np.log10(np.maximum(np.abs(h_off), 1e-12))
        ax_on.plot(w_arr, mag_on,  color=color, linestyle='-',
                   label=f'tau={tau}', linewidth=1.2)
        ax_off.plot(w_arr, mag_off, color=color, linestyle='-',
                    label=f'tau={tau}', linewidth=1.2)

    for ax, title in [(ax_on, 'Bode: ON'), (ax_off, 'Bode: OFF')]:
        ax.set_ylabel('Magnitude [dB]')
        ax.set_ylim(-12, 1)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0.25, 0.75, 1.25, 1.75])
        ax.set_xticklabels(['0.25', '0.75', '1.25', '1.75'])
        ax.text(0.97, 0.05, title, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, style='italic')
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)

    ax_off.set_xlabel('Frequency [rad/sample]')
    ax_on.legend(fontsize=7, loc='lower right', framealpha=0.7,
                 ncol=1, handlelength=1.5)


# ════════════════════════════════════════════════════════════════════════════
# Main – build figure
# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── Simulate AN-like step input ────────────────────────────────────────
    N_CFs   = 128
    T       = 150
    ONSET   = 25
    OFFSET  = 90
    dt_ms   = 1.0   # coarse dt (already downsampled)

    # Simulate with logarithmically spaced CFs (like a cochleagram)
    CFs_Hz = np.logspace(np.log10(200), np.log10(8000), N_CFs)

    # Rectangular block (step on then off) for all channels
    input_sig = make_step_signal(N_CFs, T, ONSET, OFFSET)

    # For display purposes: use a FIXED a across all CFs (like the paper figure)
    # so all frequency channels see the same response (the brown block)
    # We achieve this by passing uniform CFs -> but Willmore tau varies with CF,
    # so to match the paper's uniform-block appearance we manually run with a
    # single representative CF for display (500 Hz)
    CFs_uniform = np.full(N_CFs, 500.0)    # uniform CF → same tau everywhere
    result = apply_adaptrans(input_sig, CFs_uniform, dt_ms,
                             w=W_PAPER, K=K_PAPER, rectify=False, pad_value=0.0)

    on_out  = result[0]   # (N_CFs, T)
    off_out = result[1]

    # Also compute a log-spaced CF version for the bottom two heatmaps
    result_log = apply_adaptrans(input_sig, CFs_Hz, dt_ms,
                                 w=W_PAPER, K=None, rectify=False, pad_value=0.0)
    on_log  = result_log[0]
    off_log = result_log[1]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('white')

    # outer: left (A+C) | right (B)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1],
                              wspace=0.35, left=0.07, right=0.97,
                              top=0.96, bottom=0.07)

    # left column: A on top, C (two stacked) below
    left_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0],
        height_ratios=[1, 1, 1], hspace=0.45)

    ax_A   = fig.add_subplot(left_gs[0])
    ax_C1  = fig.add_subplot(left_gs[1])
    ax_C2  = fig.add_subplot(left_gs[2])

    # right column: trace on top, three heatmaps below
    right_gs = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer[1],
        height_ratios=[1, 1, 1, 1], hspace=0.08)

    ax_trace = fig.add_subplot(right_gs[0])
    ax_hm0   = fig.add_subplot(right_gs[1])
    ax_hm1   = fig.add_subplot(right_gs[2])
    ax_hm2   = fig.add_subplot(right_gs[3])

    # ── Draw ───────────────────────────────────────────────────────────────
    panel_A(ax_A)

    panel_B_traces(ax_trace, on_out, off_out, input_sig, ONSET, OFFSET)

    # heatmap 0: raw input (all 1s inside the block → brown)
    im0 = panel_B_heatmap(ax_hm0, input_sig, cmap='YlOrBr', diverging=False)
    plt.colorbar(im0, ax=ax_hm0, fraction=0.02, pad=0.01)

    # heatmap 1: ON output
    vmax1 = np.max(np.abs(on_log))
    norm1 = TwoSlopeNorm(vmin=-vmax1, vcenter=0, vmax=vmax1)
    im1 = ax_hm1.imshow(on_log, aspect='auto', origin='lower',
                         cmap='RdBu_r', norm=norm1, interpolation='nearest')
    ax_hm1.set_ylabel('Frequency [bins]')
    plt.colorbar(im1, ax=ax_hm1, fraction=0.02, pad=0.01)

    # heatmap 2: OFF output
    vmax2 = np.max(np.abs(off_log))
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
    im2 = ax_hm2.imshow(off_log, aspect='auto', origin='lower',
                         cmap='RdBu_r', norm=norm2, interpolation='nearest')
    ax_hm2.set_ylabel('Frequency [bins]')
    ax_hm2.set_xlabel('Time [steps]')
    plt.colorbar(im2, ax=ax_hm2, fraction=0.02, pad=0.01)

    # remove x-tick labels from top heatmaps
    for ax in [ax_hm0, ax_hm1]:
        ax.set_xticklabels([])

    panel_C(ax_C1, ax_C2)

    ax_C1.set_title('C.', loc='left', fontweight='bold')

    plt.savefig('/mnt/user-data/outputs/adaptrans_fig1.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved → /mnt/user-data/outputs/adaptrans_fig1.png")


if __name__ == '__main__':
    main()
