"""plot_torch_pipeline_stages.py
================================
Run the torch pipeline in variant-0 (notemporal) mode on a real cochlear
PSTH and compare BOLD amplitude against the saved notemporal results.

Run from project root:
    python visualization/scratch/plot_torch_pipeline_stages.py
"""

import sys, os, re
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from auditory_prf.prf_pipeline.pipeline_config import ChunkResult, PipelineConfig
from auditory_prf.prf_pipeline.torch_pipeline import (
    AuditoryPRFPipeline, build_boxcar_torch, recompute_mean_rates,
)
from auditory_prf.prf_pipeline.hrf_torch import SUBCORTICAL_PARAMS


# ── Load one real cochlear PSTH ───────────────────────────────────────────────

NPZ_PATH = (
    'models_output/dipc_8conditions_isi50ms_128ANF_wav006_cond06'
    '_fc1600hz_dur45ms_isi100ms_total20sec_numtones137.npz'
)

d           = np.load(NPZ_PATH, allow_pickle=True)
psth        = d['population_rate_psth'].astype(np.float32)
cf_hz_array = d['cf_list'].astype(np.float32)
time_axis   = d['time_axis'].astype(np.float32)

N_CF, N_TIME = psth.shape
DT_MS        = float((time_axis[1] - time_axis[0]) * 1000.0)
TOTAL_DUR    = float(time_axis[-1] * 1000.0 + DT_MS)

fname        = os.path.basename(NPZ_PATH)
TONE_DUR_MS  = float(re.search(r'dur(\d+)ms', fname).group(1))
ISI_MS       = float(re.findall(r'isi(\d+)ms', fname)[-1])
N_TONES      = int(re.search(r'numtones(\d+)', fname).group(1))
TARGET_CF    = float(re.search(r'fc(\d+)hz', fname).group(1))
cf_idx       = int(np.argmin(np.abs(cf_hz_array - TARGET_CF)))
TARGET_CF_HZ = float(cf_hz_array[cf_idx])

onsets_ms  = np.array([i * (TONE_DUR_MS + ISI_MS) for i in range(N_TONES)], dtype=np.float32)
offsets_ms = (onsets_ms + TONE_DUR_MS).astype(np.float32)

TR_S = 1.6   # match notemporal pipeline default

CHUNK = ChunkResult(
    mean_rates=np.ones(N_TONES, dtype=np.float32),
    onsets_ms=onsets_ms,
    offsets_ms=offsets_ms,
    tone_dur_ms=TONE_DUR_MS,
    total_dur_ms=TOTAL_DUR,
    dt_ms=DT_MS,
)

CONFIG = PipelineConfig(
    cf_hz=TARGET_CF_HZ,
    cf_hz_array=cf_hz_array,
    alpha=1.0,           # match notemporal alpha=1 (no sharpening)
    pref_dur_ms=45.0,    # unused in variant 0, but required by config
    sigma_dur_ms=30.0,
    w=0.8,
    tr_s=TR_S,
    hrf_params=dict(SUBCORTICAL_PARAMS),
)

psth_t = torch.from_numpy(psth)

model = AuditoryPRFPipeline(CONFIG, model_variant=0)
model.eval()

with torch.no_grad():
    bold_torch = model(psth_t, CHUNK).numpy()

tr_times = np.arange(len(bold_torch)) * TR_S / 60.0   # minutes
time_s   = time_axis

print(f"Torch BOLD: min={bold_torch.min():.2f}  max={bold_torch.max():.2f}  "
      f"shape={bold_torch.shape}")


# ── Load one saved notemporal BOLD run for comparison ─────────────────────────
# Use CF 15 run_01 from the saved output (closest available)

ref_path = 'models_output/prf_notemporal_20260521_job5665550/dipc_test_250225_01_notemporal_cf015_bold.npz'
ref      = np.load(ref_path, allow_pickle=True)
bold_ref = ref['run_01'].astype(np.float64)
tr_ref   = np.arange(len(bold_ref)) * float(ref['tr_s']) / 60.0


# ── Figure: 3 panels ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.subplots_adjust(hspace=0.45, top=0.93)

# ① PSTH heatmap
ax = axes[0]
im = ax.imshow(
    psth, aspect='auto', origin='lower',
    extent=[0, time_axis[-1], 0, N_CF - 1],
    cmap='YlOrRd', vmin=0,
)
ax.axhline(cf_idx, color='cyan', linewidth=1.2, linestyle='--',
           label=f'Target CF  {TARGET_CF_HZ:.0f} Hz')
fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85, label='Firing rate (spk/s)')
ax.set_yticks(np.linspace(0, N_CF - 1, 6, dtype=int))
ax.set_yticklabels([f'{cf_hz_array[i]:.0f}' for i in np.linspace(0, N_CF - 1, 6, dtype=int)])
ax.set_ylabel('CF (Hz)', fontsize=10)
ax.set_xlabel('Time (s)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('① Real cochlear PSTH', fontsize=11, fontweight='bold', loc='left')

# ② Torch variant-0 BOLD (single sequence)
ax = axes[1]
for on, off in zip(onsets_ms / 1000, offsets_ms / 1000):
    ax.axvspan(on, off, color='#cccccc', alpha=0.4, zorder=0)
ax.plot(np.arange(len(bold_torch)) * TR_S, bold_torch,
        color='steelblue', linewidth=1.5, label='Torch variant-0 BOLD')
ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
ax.set_xlim(0, TOTAL_DUR / 1000.0)
ax.set_ylabel('BOLD (a.u.)', fontsize=10)
ax.set_xlabel('Time (s)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title(
    f'② Torch pipeline variant-0  (no AdapTrans, no duration filter)  '
    f'α=1, CF={TARGET_CF_HZ:.0f} Hz',
    fontsize=11, fontweight='bold', loc='left',
)

# ③ Saved notemporal BOLD (full 12-min run, CF 15, run_01)
ax = axes[2]
ax.plot(tr_ref, bold_ref, color='steelblue', linewidth=1.0,
        label='Saved notemporal BOLD  (CF idx 15, run_01)')
ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
ax.set_xlim(tr_ref[0], tr_ref[-1])
ax.set_ylabel('BOLD (a.u.)', fontsize=10)
ax.set_xlabel('Time (min)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('③ Saved notemporal pipeline output  (reference)', fontsize=11, fontweight='bold', loc='left')

fig.suptitle(
    'Torch variant-0 vs saved notemporal pipeline\n'
    'Both: alpha=1, no duration filter, no AdapTrans, same HRF',
    fontsize=12, fontweight='bold',
)

out = 'visualization/scratch/torch_pipeline_stages.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
