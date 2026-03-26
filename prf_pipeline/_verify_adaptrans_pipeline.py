"""
Verification script for AdapTrans ON filter integration.
Checks:
  1. on_response.shape == (ceil(total_dur_ms),)
  2. Plot train vs on_response side-by-side (saved to file)
  3. on_response.min() >= 0  (half-wave rectification)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive — prevents plt.show() from blocking
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auditory_prf.prf_pipeline.full_pipeline_with_adaptrans import run_pipeline

EXP_NAME = "dipc_test_250225_01"
RESULTS_DIR = Path(f"./models_output/{EXP_NAME}")

# ── run pipeline ──────────────────────────────────────────────────────────────
print("Running pipeline...")
out = run_pipeline(
    exp_name=EXP_NAME,
    results_dir=RESULTS_DIR,
    alpha=2.0,
    pref_dur=200.0,
    sigma_dur=20.0,
    cf=10,
    w=0.8,
    K=None,
)
on_response = out["on_response"]
train       = out["train"]

# derive expected length directly from the returned train (avoids re-resolving the npz sub-dir)
expected_len = len(train)   # build_prf_boxcar_train uses ceil(total_dur_ms / 1.0)

# ── VERIFICATION 1: shape ─────────────────────────────────────────────────────
print("\n--- Verification 1: shape ---")
print(f"  on_response.shape : {on_response.shape}")
print(f"  expected length   : {expected_len}  (== len(train))")
assert on_response.shape == (expected_len,), (
    f"FAIL: shape mismatch — got {on_response.shape}, expected ({expected_len},)"
)
print("  PASS")

# ── VERIFICATION 2: plot train vs on_response ─────────────────────────────────
print("\n--- Verification 2: plotting ---")
t_ms = np.arange(len(train))   # 1 ms bins

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(t_ms, train, linewidth=0.9, color="steelblue")
axes[0].set_ylabel("pRF response\n(amplitude)")
axes[0].set_title("Boxcar impulse train  (steps 1–5 output)")
axes[0].set_ylim(bottom=0)

axes[1].plot(t_ms, on_response, linewidth=0.9, color="darkorange")
axes[1].set_ylabel("ON response\n(AdapTrans)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_title("AdapTrans ON filter output  (step 6–7)")
axes[1].set_ylim(bottom=0)

plt.tight_layout()
out_path = Path("output/verify_adaptrans_on_response.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"  Plot saved to: {out_path.resolve()}")
plt.close(fig)
print("  PASS")

# ── VERIFICATION 3: non-negative (half-wave rectification) ────────────────────
print("\n--- Verification 3: min >= 0 ---")
print(f"  on_response.min() = {on_response.min():.6f}")
assert on_response.min() >= 0.0, (
    f"FAIL: found negative values — min = {on_response.min():.6f}"
)
print("  PASS")

print("\nAll verifications passed.")
