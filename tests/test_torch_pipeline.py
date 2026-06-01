"""test_torch_pipeline.py
========================
Verification tests for the PyTorch auditory pRF pipeline.

Tests
-----
1. Shape test            — forward() output is (n_tr,)
2. Grad flow test        — loss.backward() populates all free parameter gradients
3. Frozen variant tests  — duration/adaptrans grads absent in variants 1 and 3
4. Module property test  — bounded param properties stay in valid ranges
5. Boxcar test           — build_boxcar_torch matches numpy build_prf_boxcar_train
6. CFSelector tests      — interpolation, bounds, grad flow through CF
7. GPU test              — .to('cuda') runs forward pass (skipped if no CUDA)

Run with:
    python tests/test_torch_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from auditory_prf.prf_pipeline.pipeline_config import ChunkResult, PipelineConfig
from auditory_prf.prf_pipeline.torch_pipeline import (
    AuditoryPRFPipeline,
    AdapTransFilter,
    CFSelector,
    DurationFilter,
    PowerLawSharpening,
    build_boxcar_torch,
    recompute_mean_rates,
)
from auditory_prf.prf_pipeline.adaptrans_onoff_filters import build_prf_boxcar_train
from auditory_prf.prf_pipeline.hrf_torch import SUBCORTICAL_PARAMS


# ── Fixtures ───────────────────────────────────────────────────────────────────

CF_HZ      = 1000.0
MIN_CF_HZ  = 125.0
MAX_CF_HZ  = 2500.0
N_CF       = 40
N_TONES    = 20
TONE_DUR   = 267.0   # ms
ISI_MS     = 67.0    # ms
DT_MS      = 1.0
TR_S       = 1.0

# Build CF array matching cochlea calc_cfs(('human') convention
_A, _k, _a = 165.4, 0.88, 2.1
_xmin = np.log10(MIN_CF_HZ / _A + _k) / _a
_xmax = np.log10(MAX_CF_HZ / _A + _k) / _a
CF_HZ_ARRAY = (_A * (10 ** (_a * np.linspace(_xmin, _xmax, N_CF)) - _k)).astype(np.float32)

onsets_ms  = np.array([i * (TONE_DUR + ISI_MS) for i in range(N_TONES)])
offsets_ms = onsets_ms + TONE_DUR
TOTAL_DUR  = float(offsets_ms[-1] + ISI_MS + 50.0)
N_TIME     = int(TOTAL_DUR / DT_MS)
N_TR       = int(TOTAL_DUR / 1000.0 / TR_S) + 1

CHUNK = ChunkResult(
    mean_rates=np.random.default_rng(0).uniform(0.5, 2.0, N_TONES).astype(np.float32),
    onsets_ms=onsets_ms.astype(np.float32),
    offsets_ms=offsets_ms.astype(np.float32),
    tone_dur_ms=TONE_DUR,
    total_dur_ms=TOTAL_DUR,
    dt_ms=DT_MS,
)

CONFIG = PipelineConfig(
    cf_hz=CF_HZ,
    cf_hz_array=CF_HZ_ARRAY,
    alpha=2.0,
    pref_dur_ms=200.0,
    sigma_dur_ms=30.0,
    w=0.8,
    tr_s=TR_S,
    hrf_params=dict(SUBCORTICAL_PARAMS),
)

def make_psth(n_cf: int = N_CF, n_time: int = N_TIME, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    psth = rng.uniform(0.1, 3.0, (n_cf, n_time)).astype(np.float32)
    return torch.from_numpy(psth)


# ── Test helpers ───────────────────────────────────────────────────────────────

def check(name: str, condition: bool, detail: str = "") -> None:
    status = "✓" if condition else "FAILED"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not condition:
        raise AssertionError(f"Test failed: {name} — {detail}")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_output_shape():
    print("\n[1] Output shape")
    model = AuditoryPRFPipeline(CONFIG, model_variant=4)
    psth  = make_psth()
    bold  = model(psth, CHUNK)
    expected_n_tr = int(TOTAL_DUR / 1000.0 / TR_S)
    check("bold is 1-D", bold.ndim == 1, f"ndim={bold.ndim}")
    check("bold length reasonable", bold.shape[0] >= expected_n_tr - 2,
          f"got {bold.shape[0]}, expected ~{expected_n_tr}")
    check("no NaNs", not torch.isnan(bold).any())
    check("no Infs", not torch.isinf(bold).any())


def test_grad_flow():
    print("\n[2] Gradient flow (variant 4)")
    model = AuditoryPRFPipeline(CONFIG, model_variant=4)
    psth  = make_psth()
    bold  = model(psth, CHUNK)
    loss  = bold.mean()
    loss.backward()

    check("CF grad exists",
          model.cf_selector._logit_x.grad is not None,
          "grad is None — gradient path from BOLD to CF is broken")
    check("alpha grad exists",
          model.sharpening._log_alpha.grad is not None,
          "grad is None — gradient path from BOLD to alpha is broken")
    check("pref_dur grad exists",
          model.duration._log_pref_dur.grad is not None)
    check("sigma_dur grad exists",
          model.duration._log_sigma_dur.grad is not None)
    # d/p params: only 2 CFs near the selected CF get gradients
    check("d_on grad exists (at least some CFs)",
          model.adaptrans.d_on.grad is not None)
    check("d_off grad exists (at least some CFs)",
          model.adaptrans.d_off.grad is not None)
    check("p grad exists (at least some CFs)",
          model.adaptrans.p.grad is not None)
    check("on_off_ratio grad exists",
          model._logit_on_off_ratio.grad is not None)


def test_frozen_variants():
    print("\n[3] Frozen parameters per variant")
    psth = make_psth()

    # Variant 1: only alpha free
    m1 = AuditoryPRFPipeline(CONFIG, model_variant=1)
    m1(psth, CHUNK).mean().backward()
    check("v1: alpha grad exists",
          m1.sharpening._log_alpha.grad is not None)
    check("v1: pref_dur grad is None",
          m1.duration._log_pref_dur.grad is None,
          "duration should be frozen in variant 1")
    check("v1: adaptrans frozen (d_on grad is None)",
          m1.adaptrans.d_on.grad is None,
          "adaptrans should be frozen in variant 1")
    check("v1: on_off_ratio grad is None",
          m1._logit_on_off_ratio.grad is None)

    # Variant 2: alpha + duration free
    m2 = AuditoryPRFPipeline(CONFIG, model_variant=2)
    m2(psth, CHUNK).mean().backward()
    check("v2: pref_dur grad exists",  m2.duration._log_pref_dur.grad is not None)
    check("v2: d_on grad is None",     m2.adaptrans.d_on.grad is None)
    check("v2: on_off_ratio grad is None", m2._logit_on_off_ratio.grad is None)

    # Variant 3: alpha + adaptrans free
    m3 = AuditoryPRFPipeline(CONFIG, model_variant=3)
    m3(psth, CHUNK).mean().backward()
    check("v3: d_on grad exists",      m3.adaptrans.d_on.grad is not None)
    check("v3: pref_dur grad is None", m3.duration._log_pref_dur.grad is None)
    check("v3: on_off_ratio grad exists", m3._logit_on_off_ratio.grad is not None)


def test_param_bounds():
    print("\n[4] Parameter bounds")
    model = AuditoryPRFPipeline(CONFIG, model_variant=4)

    with torch.no_grad():
        model.sharpening._log_alpha.fill_(10.0)
    check("alpha clamped to ALPHA_MAX",
          model.sharpening.alpha.item() <= PowerLawSharpening.ALPHA_MAX + 1e-5,
          f"alpha={model.sharpening.alpha.item():.2f}")

    with torch.no_grad():
        model.sharpening._log_alpha.fill_(-10.0)
    check("alpha clamped to ALPHA_MIN",
          model.sharpening.alpha.item() >= PowerLawSharpening.ALPHA_MIN - 1e-5,
          f"alpha={model.sharpening.alpha.item():.4f}")

    # d/p parameterisation: a = 1/(1+d²) ∈ (0,1), w = 1/(1+p²) ∈ (0,1)
    d_val = model.adaptrans.d_on[0].item()
    a_val = 1.0 / (1.0 + d_val ** 2)
    check("a_on ∈ (0,1) from d", 0.0 < a_val < 1.0, f"a={a_val:.4f}")

    p_val = model.adaptrans.p[0].item()
    w_val = 1.0 / (1.0 + p_val ** 2)
    check("w ∈ (0,1) from p", 0.0 < w_val < 1.0, f"w={w_val:.4f}")

    check("on_off_ratio in (0,1)", 0.0 < model.on_off_ratio.item() < 1.0)

    # AdapTrans shape test
    psth    = make_psth()
    sharpen = model.sharpening(psth)
    adapted = model.adaptrans(sharpen)
    check("AdapTrans output shape (2, n_cf, T)",
          adapted.shape == (2, N_CF, N_TIME),
          f"got {tuple(adapted.shape)}")


def test_boxcar_parity():
    print("\n[5] Boxcar builder parity with numpy")
    prf_responses_np = CHUNK.mean_rates.copy()
    prf_responses_t  = torch.from_numpy(prf_responses_np)

    train_np = build_prf_boxcar_train(
        prf_responses=list(prf_responses_np),
        onsets_ms=CHUNK.onsets_ms,
        offsets_ms=CHUNK.offsets_ms,
        total_dur_ms=CHUNK.total_dur_ms,
        dt_ms=CHUNK.dt_ms,
    )
    train_t = build_boxcar_torch(prf_responses_t, CHUNK).numpy()

    max_diff = np.abs(train_np - train_t).max()
    check("boxcar matches numpy", max_diff < 1e-5, f"max diff={max_diff:.2e}")
    check("boxcar shape matches", train_np.shape == train_t.shape)


def test_cf_selector():
    print("\n[6] CFSelector — interpolation, bounds, grad flow")
    sel = CFSelector(CF_HZ_ARRAY, CF_HZ)

    # Grad flows through _logit_x
    psth = make_psth()
    out  = sel(psth)
    out.mean().backward()
    check("CF grad exists", sel._logit_x.grad is not None)

    # Push to extreme logit values — cf_hz must stay in [min_cf, max_cf]
    with torch.no_grad():
        sel._logit_x.fill_(100.0)
    check("cf_hz clamped to MAX", sel.cf_hz.item() <= MAX_CF_HZ + 1.0,
          f"cf_hz={sel.cf_hz.item():.1f}")
    with torch.no_grad():
        sel._logit_x.fill_(-100.0)
    check("cf_hz clamped to MIN", sel.cf_hz.item() >= MIN_CF_HZ - 1.0,
          f"cf_hz={sel.cf_hz.item():.1f}")

    # When init_cf_hz is exactly on a grid point, output ≈ that row
    grid_cf_hz = float(CF_HZ_ARRAY[10])
    sel_exact  = CFSelector(CF_HZ_ARRAY, grid_cf_hz)
    psth2 = make_psth()
    out2  = sel_exact(psth2)
    expected = psth2[10]
    max_diff  = (out2 - expected).abs().max().item()
    check("on-grid interpolation ≈ exact row", max_diff < 1e-4,
          f"max diff={max_diff:.2e}")

    # CF grad flows through full pipeline
    model = AuditoryPRFPipeline(CONFIG, model_variant=4)
    psth3 = make_psth()
    model(psth3, CHUNK).mean().backward()
    check("CF grad in pipeline", model.cf_selector._logit_x.grad is not None)


def test_gpu():
    print("\n[7] GPU forward pass")
    if not torch.cuda.is_available():
        print("   (skipped — CUDA not available)")
        return
    model = AuditoryPRFPipeline(CONFIG, model_variant=4).to('cuda')
    psth  = make_psth().to('cuda')
    bold  = model(psth, CHUNK)
    check("output on CUDA", bold.device.type == 'cuda')
    check("no NaNs on GPU", not torch.isnan(bold).any())


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    passed = 0
    failed = 0
    tests  = [
        test_output_shape,
        test_grad_flow,
        test_frozen_variants,
        test_param_bounds,
        test_boxcar_parity,
        test_cf_selector,
        test_gpu,
    ]
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  → {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
