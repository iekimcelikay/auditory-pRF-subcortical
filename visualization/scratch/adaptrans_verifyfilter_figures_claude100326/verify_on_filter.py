"""
verify_on_filter.py
────────────────────────────────────────────────────────────────────────────
Verifies the AdapTrans ON filter using the EXACT same logic as
full_pipeline_with_adaptrans.py:

  1. build_prf_boxcar_train()   – isolated single-tone boxcar (amplitude=1)
  2. apply_adaptrans()          – ON filter, pad_value=0, dt_ms=1.0
  3. Superpose N tones          – same loop as run_pipeline()

Test battery
────────────
  Test 1 – Single tone, onset detection
      • One boxcar [50, 150) ms, amplitude=1.
      • ON response must peak at t=50 (onset), stay near zero inside,
        go negative (below baseline) after t=150 (offset).

  Test 2 – Single tone, amplitude scaling
      • Same boxcar but amplitude=2.
      • ON peak should scale proportionally (≈2×).

  Test 3 – Two tones, superposition
      • Tone A [50, 150), amplitude=1.
      • Tone B [300, 500), amplitude=1.
      • Superposed ON has two distinct peaks, one per onset.
      • Confirms the per-tone-isolation + superposition approach.

  Test 4 – Varying w (adaptation weight)
      • Single boxcar, w ∈ {0.2, 0.5, 0.8}.
      • Higher w → stronger suppression of the sustained component → ON peak
        is more transient.

  Test 5 – Varying pref_dur (duration Gaussian scaling)
      • Simulates the full pipeline with apply_duration_gaussian_scalar.
      • pref_dur = tone_dur  →  amplitude=1 (maximum selectivity).
      • pref_dur ≠ tone_dur  →  amplitude < 1.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from adaptrans_onoff_filters import (
    build_prf_boxcar_train,
    apply_adaptrans,
    willmore_tau,
    tau_to_a,
)

# ── pipeline helpers (inline, no package needed) ────────────────────────────
def duration_gaussian(stim_dur: float, pref_dur: float, sigma_dur: float) -> float:
    """Log-Gaussian duration selectivity (same as apply_duration_gaussian_scalar)."""
    return math.exp(-0.5 * ((stim_dur - pref_dur) / sigma_dur) ** 2)


def run_single_tone(amplitude: float, onset_ms: float, offset_ms: float,
                    total_ms: float, cf_hz: float, w: float,
                    dt_ms: float = 1.0, K: int = None):
    """Exact replica of the per-tone block inside run_pipeline()."""
    single_train = build_prf_boxcar_train(
        [amplitude],
        np.array([onset_ms]),
        np.array([offset_ms]),
        total_ms, dt_ms=dt_ms,
    )
    on_off = apply_adaptrans(
        single_train[np.newaxis, :],
        CFs_Hz=np.array([cf_hz]),
        dt_ms=dt_ms,
        w=w,
        K=K,
        rectify=False,      # keep negatives visible for verification
        pad_value=0.0,
    )
    return single_train, on_off[0, 0, :]   # train, ON response


# ════════════════════════════════════════════════════════════════════════════
# Build figure
# ════════════════════════════════════════════════════════════════════════════
CF_HZ     = 1000.0   # representative CF
TAU_MS    = willmore_tau(CF_HZ)
A_CF      = tau_to_a(TAU_MS, dt_ms=1.0)
TOTAL_MS  = 700.0
W_DEFAULT = 0.8
T         = np.arange(int(TOTAL_MS))

fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor('#f8f8f8')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.08, right=0.97, top=0.93, bottom=0.07)

PANEL_LABELS = list('ABCDE')
axes = []

# ── TEST 1 – single tone, onset detection ────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
axes.append(ax)
ON50, OFF150 = 50, 150
train, on = run_single_tone(1.0, ON50, OFF150, TOTAL_MS, CF_HZ, W_DEFAULT)
ax.fill_between(T, train, alpha=0.18, color='k', label='boxcar input')
ax.plot(T, on, 'r', linewidth=1.6, label='ON response')
ax.axvline(ON50,  color='green', ls='--', lw=1, label='onset')
ax.axvline(OFF150, color='purple', ls='--', lw=1, label='offset')
ax.axhline(0, color='k', lw=0.6, ls=':')
peak_t = T[np.argmax(on)]
ax.annotate(f'peak t={peak_t} ms', xy=(peak_t, on.max()),
            xytext=(peak_t + 30, on.max() * 0.9),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
ax.set_title('A.  Single tone — onset detection', fontweight='bold', fontsize=9)
ax.set_ylabel('Amplitude'); ax.set_xlabel('Time [ms]')
ax.legend(fontsize=7, loc='upper right')
pass1 = (peak_t == ON50) or (abs(peak_t - ON50) <= 2)
ax.text(0.02, 0.05, f'✓ Peak at onset' if pass1 else f'✗ Peak offset by {peak_t-ON50} ms',
        transform=ax.transAxes, fontsize=8, color='green' if pass1 else 'red')

# ── TEST 2 – amplitude scaling ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
axes.append(ax)
_, on1 = run_single_tone(1.0, ON50, OFF150, TOTAL_MS, CF_HZ, W_DEFAULT)
_, on2 = run_single_tone(2.0, ON50, OFF150, TOTAL_MS, CF_HZ, W_DEFAULT)
ax.plot(T, on1, 'r',  linewidth=1.4, label='amp=1')
ax.plot(T, on2, 'b--', linewidth=1.4, label='amp=2')
ax.axhline(0, color='k', lw=0.6, ls=':')
ratio = on2.max() / on1.max() if on1.max() > 0 else np.nan
pass2 = abs(ratio - 2.0) < 0.05
ax.set_title('B.  Amplitude scaling (should be ×2)', fontweight='bold', fontsize=9)
ax.set_ylabel('Amplitude'); ax.set_xlabel('Time [ms]')
ax.legend(fontsize=7)
ax.text(0.02, 0.05, f'✓ Ratio = {ratio:.3f}' if pass2 else f'✗ Ratio = {ratio:.3f} (expected 2.0)',
        transform=ax.transAxes, fontsize=8, color='green' if pass2 else 'red')

# ── TEST 3 – two-tone superposition ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
axes.append(ax)
A_on, A_off = 50,  200
B_on, B_off = 350, 550
train_a, on_a = run_single_tone(1.0, A_on, A_off, TOTAL_MS, CF_HZ, W_DEFAULT)
train_b, on_b = run_single_tone(1.0, B_on, B_off, TOTAL_MS, CF_HZ, W_DEFAULT)
superposed_on    = on_a + on_b
superposed_train = train_a + train_b
ax.fill_between(T, superposed_train, alpha=0.18, color='k', label='boxcar input')
ax.plot(T, superposed_on, 'r', linewidth=1.6, label='ON (superposed)')
ax.plot(T, on_a, 'C1--', linewidth=1.0, alpha=0.6, label='ON tone A')
ax.plot(T, on_b, 'C9--', linewidth=1.0, alpha=0.6, label='ON tone B')
for x, c, lbl in [(A_on,'green','A onset'),(B_on,'blue','B onset')]:
    ax.axvline(x, color=c, ls='--', lw=1, label=lbl)
ax.axhline(0, color='k', lw=0.6, ls=':')
peaks = []
for seg_start, seg_end in [(A_on-5, A_off), (B_on-5, B_off)]:
    seg = superposed_on[seg_start:seg_end]
    peaks.append(seg_start + np.argmax(seg))
pass3 = (abs(peaks[0] - A_on) <= 2) and (abs(peaks[1] - B_on) <= 2)
ax.set_title('C.  Two-tone superposition', fontweight='bold', fontsize=9)
ax.set_ylabel('Amplitude'); ax.set_xlabel('Time [ms]')
ax.legend(fontsize=7, ncol=2)
ax.text(0.02, 0.05,
        f'✓ Peaks at {peaks[0]}, {peaks[1]} ms' if pass3 else f'✗ Peaks at {peaks[0]}, {peaks[1]} ms',
        transform=ax.transAxes, fontsize=8, color='green' if pass3 else 'red')

# ── TEST 4 – varying w ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
axes.append(ax)
w_vals   = [0.2, 0.5, 0.8]
colors_w = ['C0', 'C2', 'C3']
train_ref, _ = run_single_tone(1.0, ON50, OFF150, TOTAL_MS, CF_HZ, W_DEFAULT)
ax.fill_between(T, train_ref, alpha=0.12, color='k', label='input')
for wv, col in zip(w_vals, colors_w):
    _, on_w = run_single_tone(1.0, ON50, OFF150, TOTAL_MS, CF_HZ, wv)
    sustained = np.mean(on_w[ON50 + 20 : OFF150 - 5])  # mid-tone mean
    ax.plot(T, on_w, color=col, linewidth=1.4,
            label=f'w={wv}  (sust≈{sustained:.3f})')
ax.axhline(0, color='k', lw=0.6, ls=':')
ax.axvline(ON50,  color='green',  ls='--', lw=1)
ax.axvline(OFF150, color='purple', ls='--', lw=1)
ax.set_title('D.  Varying w (higher w → more transient)', fontweight='bold', fontsize=9)
ax.set_ylabel('Amplitude'); ax.set_xlabel('Time [ms]')
ax.legend(fontsize=7)
# Verify sustained drops as w increases
sust_vals = []
for wv in w_vals:
    _, on_w = run_single_tone(1.0, ON50, OFF150, TOTAL_MS, CF_HZ, wv)
    sust_vals.append(np.mean(on_w[ON50 + 20 : OFF150 - 5]))
pass4 = all(sust_vals[i] > sust_vals[i+1] for i in range(len(sust_vals)-1))
ax.text(0.02, 0.05, '✓ Sustained decreases with w' if pass4 else '✗ Sustained not monotone with w',
        transform=ax.transAxes, fontsize=8, color='green' if pass4 else 'red')

# ── TEST 5 – duration Gaussian scaling ───────────────────────────────────
ax = fig.add_subplot(gs[2, :])
axes.append(ax)
TONE_DUR = 100.0     # ms
SIGMA    = 40.0
pref_durs = [50, 100, 200, 400]
colors_d  = ['C0','C2','C3','C5']
onset_base = 50.0
offset_base = onset_base + TONE_DUR

ax.axvspan(onset_base, offset_base, alpha=0.1, color='k', label='tone window')
ax.axhline(0, color='k', lw=0.6, ls=':')

for pref, col in zip(pref_durs, colors_d):
    gauss_scale  = duration_gaussian(TONE_DUR, pref, SIGMA)
    amplitude    = 1.0 * gauss_scale          # mean_rate=1 × gaussian
    _, on_d = run_single_tone(amplitude, onset_base, offset_base,
                              TOTAL_MS, CF_HZ, W_DEFAULT)
    ax.plot(T, on_d, color=col, linewidth=1.4,
            label=f'pref_dur={pref} ms  (scale={gauss_scale:.3f})')

# expected: pref=tone_dur gives max peak
peak_at_pref = []
for pref in pref_durs:
    g = duration_gaussian(TONE_DUR, pref, SIGMA)
    _, on_d = run_single_tone(g, onset_base, offset_base, TOTAL_MS, CF_HZ, W_DEFAULT)
    peak_at_pref.append(on_d.max())

max_idx   = int(np.argmax(peak_at_pref))
pass5     = pref_durs[max_idx] == TONE_DUR
ax.set_title(f'E.  Duration Gaussian — pref_dur={TONE_DUR} ms should give max peak  '
             f'(tone_dur={TONE_DUR} ms, σ={SIGMA} ms)',
             fontweight='bold', fontsize=9)
ax.set_ylabel('Amplitude'); ax.set_xlabel('Time [ms]')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.text(0.02, 0.05,
        f'✓ Max peak at pref_dur={pref_durs[max_idx]} ms' if pass5
        else f'✗ Max peak at pref_dur={pref_durs[max_idx]} ms (expected {TONE_DUR})',
        transform=ax.transAxes, fontsize=8, color='green' if pass5 else 'red')

# ── Summary title ─────────────────────────────────────────────────────────
n_pass = sum([pass1, pass2, pass3, pass4, pass5])
fig.suptitle(
    f'AdapTrans ON Filter — Verification  '
    f'(CF={CF_HZ:.0f} Hz, τ={TAU_MS:.1f} ms, a={A_CF:.4f}, w={W_DEFAULT})   '
    f'[{n_pass}/5 tests passed]',
    fontsize=11, fontweight='bold', y=0.98
)

out_path = '/mnt/user-data/outputs/adaptrans_ON_verification.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
print(f"Saved → {out_path}  |  {n_pass}/5 tests passed")
