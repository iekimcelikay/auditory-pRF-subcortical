"""
generate_presentation_figures.py
─────────────────────────────────────────────────────────────────────────────
Generates the three panel figures embedded in the AdapTrans presentation:

  fig_kernel.png           — ON kernel stem plot (K=10, w=0.5, a=0.6)
  fig_step_response.png    — Step response + effect of w
  fig_duration_gaussian.png — Duration Gaussian bell curves
  fig_pad_value.png        — pad_value=None vs pad_value=0.0 comparison

Run from the same directory as adaptrans_onoff_filters.py:
  python generate_presentation_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

from adaptrans_onoff_filters import (
    build_ON_kernel,
    build_prf_boxcar_train,
    apply_adaptrans,
    willmore_tau,
    tau_to_a,
)

# ── shared settings ──────────────────────────────────────────────────────────
CF_HZ   = 1000.0
TAU_MS  = willmore_tau(CF_HZ)
A_CF    = tau_to_a(TAU_MS, 1.0)
TOTAL   = 400
T       = np.arange(TOTAL)
ON_MS   = 50
OFF_MS  = 200

def run_tone(amp, on, off, total, cf, w_val, K=None):
    """Exact replica of the per-tone block in run_pipeline()."""
    tr = build_prf_boxcar_train(
        [amp], np.array([on]), np.array([off]), float(total), 1.0
    )
    res = apply_adaptrans(
        tr[np.newaxis, :],
        CFs_Hz=np.array([cf]),
        dt_ms=1.0,
        w=w_val,
        K=K,
        rectify=False,
        pad_value=0.0,
    )
    return tr, res[0, 0, :]


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — ON kernel stem plot
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="white")

K, w, a = 10, 0.5, 0.6
kern = build_ON_kernel(a, w, K)
x    = np.arange(K)

ml, sl, bl = ax.stem(x, kern, linefmt="r-", markerfmt="ro", basefmt="k-")
plt.setp(sl, linewidth=1.5)
plt.setp(ml, markersize=6)

ax.axhline(0, color="k", lw=0.8)
ax.annotate(
    "+1 (current)",
    xy=(K - 1, kern[-1]), xytext=(K - 3, 0.8),
    arrowprops=dict(arrowstyle="->", color="darkred"),
    fontsize=9, color="darkred",
)
ax.annotate(
    "−w×past\n(exponential decay)",
    xy=(2, kern[2]), xytext=(4, -0.35),
    arrowprops=dict(arrowstyle="->", color="navy"),
    fontsize=9, color="navy",
)
ax.set_xlabel("← past                    present →", fontsize=9)
ax.set_ylabel("Weight", fontsize=9)
ax.set_title(f"ON Kernel  (K={K}, w={w}, a={a})", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_kernel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → fig_kernel.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Step response + effect of w
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor="white")

# left: single step response
tr, on_resp = run_tone(1.0, ON_MS, OFF_MS, TOTAL, CF_HZ, 0.8)
ax = axes[0]
ax.fill_between(T, tr, alpha=0.15, color="k")
ax.plot(T, tr,      "k-", lw=1.2, label="Input (boxcar)")
ax.plot(T, on_resp, "r-", lw=2.0, label="ON response")
ax.axvline(ON_MS,  color="green",  ls="--", lw=1.2, label="Onset")
ax.axvline(OFF_MS, color="purple", ls="--", lw=1.2, label="Offset")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Amplitude")
ax.set_title("Step response  (CF=1kHz, w=0.8)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

# right: varying w
ax = axes[1]
for wv, col, ls in [(0.2, "C0", "-"), (0.5, "C2", "--"), (0.8, "C3", ":")]:
    _, on_w = run_tone(1.0, ON_MS, OFF_MS, TOTAL, CF_HZ, wv)
    ax.plot(T, on_w, color=col, lw=1.8, ls=ls, label=f"w={wv}")
ax.fill_between(T, tr, alpha=0.1, color="k")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.axvline(ON_MS,  color="green",  ls="--", lw=1.0, alpha=0.6)
ax.axvline(OFF_MS, color="purple", ls="--", lw=1.0, alpha=0.6)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Amplitude")
ax.set_title("Effect of w  (higher w → more transient)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig_step_response.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → fig_step_response.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Duration Gaussian
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor="white")

durs = np.linspace(10, 600, 500)
sigma = 80
for pref, col in [(50, "C0"), (200, "C2"), (400, "C3")]:
    g = np.exp(-0.5 * ((durs - pref) / sigma) ** 2)
    ax.plot(durs, g, color=col, lw=2, label=f"pref_dur={pref} ms")

ax.axvline(200, color="gray", ls=":", lw=1)
ax.set_xlabel("Tone duration [ms]")
ax.set_ylabel("Gaussian weight")
ax.set_title(f"Duration Gaussian  (σ={sigma} ms)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("fig_duration_gaussian.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → fig_duration_gaussian.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — pad_value comparison
# ════════════════════════════════════════════════════════════════════════════
K_PAD   = 80
TOTAL_P = 300
ON_P    = 0      # tone starts immediately at t=0 — worst case for pad_value
OFF_P   = 100

tr_p = build_prf_boxcar_train(
    [1.0], np.array([float(ON_P)]), np.array([float(OFF_P)]), float(TOTAL_P), 1.0
)

res_none = apply_adaptrans(
    tr_p[np.newaxis, :], np.array([CF_HZ]),
    dt_ms=1.0, w=0.8, K=K_PAD, rectify=False, pad_value=None,
)
res_zero = apply_adaptrans(
    tr_p[np.newaxis, :], np.array([CF_HZ]),
    dt_ms=1.0, w=0.8, K=K_PAD, rectify=False, pad_value=0.0,
)

T_p = np.arange(TOTAL_P)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor="white")

for ax, res, title, note, note_col in [
    (axes[0], res_none[0, 0, :],
     "pad_value=None\n(replicate first sample)", "✗ onset suppressed", "red"),
    (axes[1], res_zero[0, 0, :],
     "pad_value=0.0\n(assume silence before)",   "✓ onset detected",   "green"),
]:
    ax.fill_between(T_p, tr_p, alpha=0.15, color="k", label="input")
    ax.plot(T_p, res, "r-", lw=2, label="ON response")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.axvline(ON_P, color="green", ls="--", lw=1.2)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(
        0.98, 0.95, note,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=note_col, fontweight="bold",
    )
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig_pad_value.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → fig_pad_value.png")

print("\nAll figures saved. Run make_presentation.js to embed them in the deck.")
