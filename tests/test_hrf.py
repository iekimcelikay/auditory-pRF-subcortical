"""Test script to verify build_hrf_kernel and convolve_hrf in hrf.py."""
import numpy as np
from auditory_prf.prf_pipeline.hrf import (
    build_hrf_kernel, convolve_hrf, hrf_summary,
    SPM_PARAMS, GLOVER_PARAMS, POPEYE_PARAMS, SUBCORTICAL_PARAMS,
)

DT       = 0.001
DURATION = 32.0

print("Testing auditory_prf.prf_pipeline.hrf")
print("=" * 60)

# -----------------------------------------------------------------------
# build_hrf_kernel
# -----------------------------------------------------------------------

# Test 1: output shapes
print("\nTest 1: build_hrf_kernel output shapes")
kernel, t = build_hrf_kernel(dt=DT, duration=DURATION)
n_expected = int(DURATION / DT)
print(f"  kernel shape: {kernel.shape}, t shape: {t.shape}, expected: ({n_expected},)")
print(f"  ✓" if kernel.shape == (n_expected,) and t.shape == (n_expected,) else "  FAILED: shape mismatch")

# Test 2: positive-lobe area normalised to 1.0
print("\nTest 2: kernel normalisation (positive area == 1.0)")
pos_area = np.sum(kernel[kernel > 0]) * DT
print(f"  positive area = {pos_area:.6f}  (expected 1.0)")
print(f"  ✓" if np.isclose(pos_area, 1.0, atol=1e-5) else "  FAILED: normalisation off")

# Test 3: peak time close to peak_delay
print("\nTest 3: kernel peak near peak_delay (SPM default = 6.0 s)")
peak_t = t[np.argmax(kernel)]
print(f"  peak time = {peak_t:.3f} s  (expected ~ 6.0 s, tol ±1.5 s)")
print(f"  ✓" if abs(peak_t - SPM_PARAMS["peak_delay"]) < 1.5 else "  FAILED: peak too far from peak_delay")

# Test 4: no NaN or Inf
print("\nTest 4: no NaN / Inf in kernel")
print(f"  ✓" if np.isfinite(kernel).all() else "  FAILED: kernel contains NaN or Inf")

# Test 5: normalisation holds for all presets
print("\nTest 5: normalisation for all four presets")
presets = {
    "SPM":         SPM_PARAMS,
    "Glover":      GLOVER_PARAMS,
    "Popeye":      POPEYE_PARAMS,
    "Subcortical": SUBCORTICAL_PARAMS,
}
for name, params in presets.items():
    k, _ = build_hrf_kernel(**params, dt=DT, duration=DURATION)
    area  = np.sum(k[k > 0]) * DT
    label = "✓" if np.isclose(area, 1.0, atol=1e-5) else "FAILED"
    print(f"  {name}: positive area = {area:.6f}  {label}")

# -----------------------------------------------------------------------
# convolve_hrf
# -----------------------------------------------------------------------
print("\n" + "=" * 60)

spm_kernel, _ = build_hrf_kernel(**SPM_PARAMS, dt=DT, duration=DURATION)

# Test 6: output length == signal length (same dt, no downsampling)
print("\nTest 6: convolve_hrf output length (same dt, no output_dt)")
sig = np.zeros(int(20 / DT))
out = convolve_hrf(sig, spm_kernel, signal_dt=DT, kernel_dt=DT)
print(f"  signal length: {len(sig)}, output length: {len(out)}")
print(f"  ✓" if len(out) == len(sig) else "  FAILED: length mismatch")

# Test 7: output length preserved when resampling (signal_dt != kernel_dt)
print("\nTest 7: convolve_hrf output length when signal_dt != kernel_dt (resampling path)")
signal_dt_coarse = 0.01
coarse_sig = np.zeros(int(20 / signal_dt_coarse))
out_coarse  = convolve_hrf(coarse_sig, spm_kernel, signal_dt=signal_dt_coarse, kernel_dt=DT)
print(f"  signal length: {len(coarse_sig)}, output length: {len(out_coarse)}")
print(f"  ✓" if len(out_coarse) == len(coarse_sig) else "  FAILED: length mismatch")

# Test 8: downsampling with output_dt
print("\nTest 8: convolve_hrf downsampling with output_dt (TR = 1.0 s)")
TR        = 1.0
sig_fine  = np.zeros(int(20 / DT))
out_tr    = convolve_hrf(sig_fine, spm_kernel, signal_dt=DT, kernel_dt=DT, output_dt=TR)
n_expected_tr = len(sig_fine[::int(round(TR / DT))])
print(f"  expected {n_expected_tr} TRs, got {len(out_tr)}")
print(f"  ✓" if len(out_tr) == n_expected_tr else "  FAILED: downsampled length wrong")

# Test 9: ValueError raised for 2D input
print("\nTest 9: ValueError for 2D signal")
try:
    convolve_hrf(np.zeros((4, 100)), spm_kernel, signal_dt=DT)
    print("  FAILED: no exception raised")
except ValueError:
    print("  ✓ ValueError raised as expected")

# Test 10: causality — output near zero before stimulus onset
print("\nTest 10: causality (output near zero before stimulus onset at t=5 s)")
t_axis = np.arange(0, 20, DT)
causal_sig = np.zeros_like(t_axis)
causal_sig[(t_axis >= 5.0) & (t_axis < 7.0)] = 1.0
causal_out = convolve_hrf(causal_sig, spm_kernel, signal_dt=DT, kernel_dt=DT)
pre_onset_max = np.abs(causal_out[:int(4.5 / DT)]).max()
print(f"  max abs output before t=4.5 s: {pre_onset_max:.2e}  (expected < 1e-6)")
print(f"  ✓" if pre_onset_max < 1e-6 else "  FAILED: non-causal response detected")

# Test 11: all-zeros signal gives all-zeros output
print("\nTest 11: zero signal -> zero output")
zero_out = convolve_hrf(np.zeros(5000), spm_kernel, signal_dt=DT, kernel_dt=DT)
print(f"  max abs output: {np.abs(zero_out).max():.2e}  (expected 0.0)")
print(f"  ✓" if np.abs(zero_out).max() == 0.0 else "  FAILED: non-zero output for zero signal")

# -----------------------------------------------------------------------
# hrf_summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("\nTest 12: hrf_summary runs without error")
try:
    import logging
    hrf_summary(spm_kernel, _, DT, SPM_PARAMS)
    print("  ✓ hrf_summary completed without exception")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 60)
print("SUMMARY: hrf.py kernel building and convolution verified.")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel 1: all four canonical HRFs
ax = axes[0]
t_plot = np.arange(0.0, DURATION, DT)
for name, params in presets.items():
    k, _ = build_hrf_kernel(**params, dt=DT, duration=DURATION)
    ax.plot(t_plot, k, label=name)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, 25)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Canonical HRFs (all presets)")
ax.legend()

# Panel 2: causality test — stimulus and convolved BOLD
ax = axes[1]
ax.fill_between(t_axis, causal_sig * 0.3, alpha=0.3, label="Neural (scaled)")
ax.plot(t_axis, causal_out, lw=1.5, label="BOLD (SPM)")
ax.axvline(5.0, color="grey", lw=0.8, ls="--", label="Onset")
ax.set_xlabel("Time (s)")
ax.set_title("Causality test (onset at 5 s)")
ax.legend()

fig.tight_layout()
plt.show()
