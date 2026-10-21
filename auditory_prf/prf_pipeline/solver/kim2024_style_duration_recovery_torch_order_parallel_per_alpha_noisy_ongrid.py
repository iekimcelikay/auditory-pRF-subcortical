"""kim2024_style_duration_recovery_torch_order_parallel_per_alpha_noisy_ongrid.py
==============================================================================
ON-GRID ground truths with the noise sweep, now that noise-free on-grid recovery
is exact. Wraps
``kim2024_style_duration_recovery_torch_order_parallel_per_alpha.py`` and does
NOT modify it -- overrides module globals in place before calling its own
(otherwise unmodified) ``main()``.

What changes relative to the noisy script:
  - ``GT_MODE = "on_grid"``: ground truths are random candidate-grid rows, so any
    recovery error is due to noise / identifiability, not grid quantization.
  - PmNoise jitter is exposed as ``NOISE_JITTER`` (frequency, amplitude, phase
    SD as a fraction of nominal, applied to the periodic noise components).
    It is [0, 0, 0] for the first run; set it to [0.1, 0.1, 0] for the jitter run.
  - No noise-free ``"none"`` row is run here (it already exists in the nonoise
    results).
``N_VOXELS`` is overridden to 300 here (default 100). Everything else (noise
levels low/mid/high, ``N_NOISE_REPEATS``, the 117-pair duration-scaled grid, the 10 design seeds via
``SLURM_ARRAY_TASK_ID``) is inherited unchanged from the noisy script.

Results go to their own folder, so the noise-free and off-grid runs are untouched.

Usage (from project root, request multiple CPUs via SLURM -- see
slurm_scripts/batch_kim2024_style_duration_recovery_torch_order_per_alpha_noisy_ongrid.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha_noisy_ongrid
"""

from functools import partial
from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order as b
import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha as p

# PmNoise jitter [frequency, amplitude, phase] as fraction of nominal.
# Step 1: no jitter (same as the preset default). Step 2: set to [0.1, 0.1, 0.0].
NOISE_JITTER = [0.0, 0.0, 0.0]

p.GT_MODE = "on_grid"
# Number of ground-truth voxels (inherited default is 100). ``main()`` reads
# ``p.N_VOXELS`` for ground-truth drawing, the progress print and the plot.
N_VOXELS = 300
p.N_VOXELS = N_VOXELS
# draw_noisy_measurement was imported BY VALUE into ``p``; wrap it there so every
# worker (forked after this patch) draws jittered noise.
p.draw_noisy_measurement = partial(p.draw_noisy_measurement, jitter=NOISE_JITTER)

OUTPUT_DIR = Path("kim2024_style_duration_recovery_torch_order_noisy_ongrid_results")
b.OUTPUT_DIR = OUTPUT_DIR
b.RUN_DIR    = OUTPUT_DIR / b.RUN_TIMESTAMP
p.RUN_DIR    = b.RUN_DIR


if __name__ == "__main__":
    print(f"[noisy_ongrid] GT_MODE={p.GT_MODE}, noise levels={p.NOISE_VOXELS}, "
          f"N_VOXELS={p.N_VOXELS}, N_NOISE_REPEATS={p.N_NOISE_REPEATS:,}, jitter={NOISE_JITTER}")
    p.main()
