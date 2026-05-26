"""Test script to verify assemble_run_bold in run_assembly.py.

Uses synthetic boxcar trains so no cochlear simulation or WAV files are
needed.  Run time is < 5 s at 60-sec mini-run resolution.
"""
import numpy as np
from auditory_prf.prf_pipeline.hrf import build_hrf_kernel, SUBCORTICAL_PARAMS
from auditory_prf.prf_pipeline.run_assembly import assemble_run_bold

# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------
def make_boxcar(n_tones, tone_dur_ms, isi_ms, amp=50.0):
    """Build a multi-tone pRF-weighted boxcar train (1 ms resolution).

    Total = n_tones * tone_dur_ms + (n_tones - 1) * isi_ms  (no trailing ISI).
    """
    total = int(n_tones * tone_dur_ms + (n_tones - 1) * isi_ms)
    train = np.zeros(total)
    for i in range(n_tones):
        on = int(i * (tone_dur_ms + isi_ms))
        train[on : on + int(tone_dur_ms)] = amp
    return train

DT_S        = 1e-3
TR_S        = 1.0
TOTAL_RUN_S = 60.0
CF_HZ       = 125.0

hrf_kernel, _ = build_hrf_kernel(**SUBCORTICAL_PARAMS, dt=DT_S, duration=32.0)

# Two sequences: 36 tones × 200 ms + 35 × 80 ms ISI = 10,000 ms exactly (no trailing ISI)
seq_a = make_boxcar(36, 200, 80, amp=50.0)
seq_b = make_boxcar(36, 200, 80, amp=30.0)

per_seq = {
    "seq_a": {"train": seq_a},
    "seq_b": {"train": seq_b},
}

# Run design: spaced (10 s gap between seq_a end and seq_b start)
# seq_a:  5–15 s | seq_b: 25–35 s | null: 40 s | rep seq_a: 45–55 s
RUN_DESIGN = [
    ("seq_a", 5.0),
    ("seq_b", 25.0),
    (None,    40.0),   # null trial
    ("seq_a", 45.0),   # repetition of seq_a
]

result = assemble_run_bold(
    per_seq         = per_seq,
    run_design      = RUN_DESIGN,
    total_run_dur_s = TOTAL_RUN_S,
    hrf_kernel      = hrf_kernel,
    cf_hz           = CF_HZ,
    tr_s            = TR_S,
)

print("Testing auditory_prf.prf_pipeline.run_assembly")
print("=" * 60)

# -----------------------------------------------------------------------
# Test 1: output shapes
# -----------------------------------------------------------------------
print("\nTest 1: output shapes")
n_tr_expected  = int(TOTAL_RUN_S / TR_S)
n_1ms_expected = int(round(TOTAL_RUN_S / DT_S))
shapes_ok = (
    result["bold_on"].shape       == (n_tr_expected,)  and
    result["bold_off"].shape      == (n_tr_expected,)  and
    result["bold_combined"].shape == (n_tr_expected,)  and
    result["t_tr"].shape          == (n_tr_expected,)  and
    result["full_train"].shape    == (n_1ms_expected,) and
    result["on_response"].shape   == (n_1ms_expected,) and
    result["off_response"].shape  == (n_1ms_expected,)
)
print(f"  bold_on:    {result['bold_on'].shape}  (expected ({n_tr_expected},))")
print(f"  full_train: {result['full_train'].shape}  (expected ({n_1ms_expected},))")
print(f"  ✓" if shapes_ok else "  FAILED: shape mismatch")

# -----------------------------------------------------------------------
# Test 2: bold_combined == bold_on + bold_off exactly
# -----------------------------------------------------------------------
print("\nTest 2: bold_combined == bold_on + bold_off")
diff = np.abs(result["bold_combined"] - (result["bold_on"] + result["bold_off"])).max()
print(f"  max diff: {diff:.2e}  (expected 0.0)")
print(f"  ✓" if diff == 0.0 else "  FAILED")

# -----------------------------------------------------------------------
# Test 3: t_tr time axis
# -----------------------------------------------------------------------
print("\nTest 3: t_tr axis values")
t_expected = np.arange(n_tr_expected) * TR_S
print(f"  t_tr[0]={result['t_tr'][0]:.1f}  t_tr[-1]={result['t_tr'][-1]:.1f}  (expected 0.0 … {(n_tr_expected-1)*TR_S:.1f})")
print(f"  ✓" if np.allclose(result["t_tr"], t_expected) else "  FAILED: t_tr wrong")

# -----------------------------------------------------------------------
# Test 4: onset placement — full_train correct at seq_a onset (5.0 s)
# -----------------------------------------------------------------------
print("\nTest 4: full_train has correct amplitude at seq_a onset (5.0 s)")
onset_sample  = int(5.0 / DT_S)
window        = result["full_train"][onset_sample : onset_sample + len(seq_a)]
placed_max    = window.max()
print(f"  max in window: {placed_max:.2f}  (expected {seq_a.max():.2f})")
print(f"  ✓" if np.isclose(placed_max, seq_a.max()) else "  FAILED: train not placed correctly")

# -----------------------------------------------------------------------
# Test 5: null trial — full_train zero in null window (15.0–20.0 s)
# -----------------------------------------------------------------------
print("\nTest 5: full_train is zero between seq_b end (35 s) and rep onset (45 s)")
null_start = int(36.0 / DT_S)
null_end   = int(44.0 / DT_S)
null_max   = np.abs(result["full_train"][null_start:null_end]).max()
print(f"  max abs in null window: {null_max:.2e}  (expected 0.0)")
print(f"  ✓" if null_max == 0.0 else "  FAILED: non-zero in null window")

# -----------------------------------------------------------------------
# Test 6: repetition — full_train non-zero at second seq_a onset (20.0 s)
# -----------------------------------------------------------------------
print("\nTest 6: full_train correct at seq_a repetition (45.0 s)")
rep_onset = int(45.0 / DT_S)
rep_max   = result["full_train"][rep_onset : rep_onset + len(seq_a)].max()
print(f"  max at rep onset: {rep_max:.2f}  (expected {seq_a.max():.2f})")
print(f"  ✓" if np.isclose(rep_max, seq_a.max()) else "  FAILED: repetition not placed")

# -----------------------------------------------------------------------
# Test 7: causality — BOLD near zero before first stimulus onset
# -----------------------------------------------------------------------
print("\nTest 7: BOLD near zero before first onset (first 4 TRs)")
pre_max = np.abs(result["bold_combined"][:4]).max()
print(f"  max abs BOLD before t=4 TR: {pre_max:.2e}  (expected < 1e-6)")
print(f"  ✓" if pre_max < 1e-6 else "  FAILED: non-causal response")

# -----------------------------------------------------------------------
# Test 8: different amplitudes produce different full_train values
# -----------------------------------------------------------------------
print("\nTest 8: seq_b (amp=30) places lower amplitude than seq_a (amp=50)")
seq_b_onset = int(25.0 / DT_S)
b_max = result["full_train"][seq_b_onset : seq_b_onset + len(seq_b)].max()
a_max = result["full_train"][onset_sample : onset_sample + len(seq_a)].max()
print(f"  seq_a max in train: {a_max:.2f}   seq_b max in train: {b_max:.2f}")
print(f"  ✓" if b_max < a_max else "  FAILED: amplitudes not differentiated")

# -----------------------------------------------------------------------
# Test 9: apply_adaptrans_flag=False — on_response == full_train
# -----------------------------------------------------------------------
print("\nTest 9: apply_adaptrans_flag=False — on_response == full_train, off_response == 0")
result_noat = assemble_run_bold(
    per_seq=per_seq, run_design=RUN_DESIGN,
    total_run_dur_s=TOTAL_RUN_S, hrf_kernel=hrf_kernel,
    cf_hz=CF_HZ, tr_s=TR_S, apply_adaptrans_flag=False,
)
on_eq_train  = np.allclose(result_noat["on_response"], result_noat["full_train"])
off_eq_zeros = np.allclose(result_noat["off_response"], 0.0)
print(f"  on_response == full_train:  {on_eq_train}")
print(f"  off_response all zeros:     {off_eq_zeros}")
print(f"  ✓" if on_eq_train and off_eq_zeros else "  FAILED")

# -----------------------------------------------------------------------
# Test 10: missing seq_id handled without crash
# -----------------------------------------------------------------------
print("\nTest 10: missing seq_id in run_design handled gracefully (no exception)")
try:
    result_miss = assemble_run_bold(
        per_seq=per_seq,
        run_design=[("nonexistent_seq", 5.0)],
        total_run_dur_s=TOTAL_RUN_S, hrf_kernel=hrf_kernel,
        cf_hz=CF_HZ, tr_s=TR_S,
    )
    all_zero = np.all(result_miss["full_train"] == 0.0)
    print(f"  no exception raised, full_train all zeros: {all_zero}")
    print(f"  ✓" if all_zero else "  FAILED: full_train not zero")
except Exception as e:
    print(f"  FAILED: raised {e}")

print("\n" + "=" * 60)
print("SUMMARY: onset placement, null silence, repetition, causality,")
print("         amplitude differentiation, and no-AdapTrans path verified.")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import apply_adaptrans as _at

SEQ_DUR_S = len(seq_a) * DT_S   # 10.0 s

# --- Back-to-back run: seq_b starts immediately after seq_a ends
RUN_DESIGN_B2B = [
    ("seq_a", 5.0),
    ("seq_b", 5.0 + SEQ_DUR_S),   # 15.0 s — no gap
    (None,    30.0),
    ("seq_a", 40.0),
]
result_b2b = assemble_run_bold(
    per_seq=per_seq, run_design=RUN_DESIGN_B2B,
    total_run_dur_s=TOTAL_RUN_S, hrf_kernel=hrf_kernel,
    cf_hz=CF_HZ, tr_s=TR_S,
)

# --- Within-sequence with rectify=True: full-train vs per-tone sum now differ
# Linear case (rectify=False): identical by superposition — not shown
# Non-linear case (rectify=True): negative sidelobes clipped before summing in per-tone,
#   but after summing in full-train → forward masking effect visible
ft_out_rect        = _at(seq_a[np.newaxis, :], CFs_Hz=np.array([CF_HZ]), dt_ms=1.0,
                         pad_value=0.0, rectify=True)
seq_a_full_on_rect = ft_out_rect[0, 0, :]

seq_a_pertone_rect = np.zeros(len(seq_a))
for i in range(36):
    single = np.zeros(len(seq_a))
    on_s   = int(i * (200 + 80))
    single[on_s : on_s + 200] = 50.0
    out_s = _at(single[np.newaxis, :], CFs_Hz=np.array([CF_HZ]), dt_ms=1.0,
                pad_value=0.0, rectify=True)
    seq_a_pertone_rect += out_s[0, 0, :]

t_1ms = np.arange(len(result["full_train"])) * DT_S
t_seq = np.arange(len(seq_a)) * DT_S

fig, axes = plt.subplots(4, 1, figsize=(13, 13))

# Panel 1: run-level ON — spaced vs back-to-back
ax = axes[0]
ax.plot(t_1ms, result["on_response"],     color="steelblue", lw=0.9, label="Spaced (10 s gap)")
ax.plot(t_1ms, result_b2b["on_response"], color="crimson",   lw=0.9, label="Back-to-back (0 s gap)", alpha=0.85)
for _, onset_s in RUN_DESIGN:
    ax.axvline(onset_s, color="steelblue", lw=0.7, ls=":", alpha=0.5)
for _, onset_s in RUN_DESIGN_B2B:
    ax.axvline(onset_s, color="crimson",   lw=0.7, ls=":", alpha=0.5)
ax.set_xlim(3, 55)
ax.set_ylabel("ON (spk/s)")
ax.set_title("Run-level AdapTrans ON: spaced vs back-to-back")
ax.legend(fontsize=8)

# Panel 2: zoom into seq_b window — onset suppression in b2b case
ax = axes[1]
ax.plot(t_1ms, result["on_response"],     color="steelblue", lw=1.2, label="Spaced  (seq_b at 25 s)")
ax.plot(t_1ms, result_b2b["on_response"], color="crimson",   lw=1.2, label="Back-to-back  (seq_b at 15 s)")
ax.axvline(25.0,           color="steelblue", lw=1.0, ls="--")
ax.axvline(5.0 + SEQ_DUR_S, color="crimson",  lw=1.0, ls="--")
ax.set_xlim(13, 28)
ax.set_ylabel("ON (spk/s)")
ax.set_title("Zoom: seq_b first onset — b2b (red) suppressed vs spaced (blue)")
ax.legend(fontsize=8)

# Panel 3: within seq_a with rectify=True — full-train vs per-tone sum now differ
ax = axes[2]
ax.plot(t_seq, seq_a_pertone_rect, color="grey",    lw=1.2, ls="--", label="Per-tone sum (rectify=True)")
ax.plot(t_seq, seq_a_full_on_rect, color="crimson", lw=1.4,          label="Full-train (rectify=True)")
for i in range(36):
    on_s  = i * (200 + 80) * DT_S
    off_s = on_s + 200 * DT_S
    ax.axvspan(on_s, off_s, alpha=0.06, color="steelblue")
ax.set_xlim(0, SEQ_DUR_S)
ax.set_ylabel("ON (spk/s)")
ax.set_title("Within seq_a (rectify=True): full-train onset suppression vs per-tone sum")
ax.legend(fontsize=8)

# Panel 4: BOLD combined — spaced vs back-to-back
ax = axes[3]
ax.plot(result["t_tr"],     result["bold_combined"],     "o-", ms=4, color="steelblue", label="Spaced")
ax.plot(result_b2b["t_tr"], result_b2b["bold_combined"], "o-", ms=4, color="crimson",   label="Back-to-back")
ax.set_xlabel("Time (s)")
ax.set_ylabel("BOLD (a.u.)")
ax.set_title(f"BOLD combined at TR={TR_S:.1f} s")
ax.legend(fontsize=8)

fig.tight_layout()
plt.show()
