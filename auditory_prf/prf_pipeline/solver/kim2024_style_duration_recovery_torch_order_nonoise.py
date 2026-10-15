"""kim2024_style_duration_recovery_torch_order_nonoise.py
=========================================================
Noise-FREE run of kim2024_style_duration_recovery_torch_order.py's pipeline
(torch order, population-pool normalization), to test duration-model recovery
before any noise is added. Does NOT modify
kim2024_style_duration_recovery_torch_order.py -- overrides its module globals
in place before calling its own (otherwise unmodified) main().

With no noise the measurement is deterministic, so one "repeat" per voxel is
enough. Recovery should match the ground truth up to grid resolution; any
remaining bias comes from the model itself (normalization, silence handling,
off-grid ground truths), not from noise.

The grids are NOT shrunk here (unlike the ``_debug`` variant), because a
2-value grid would make off-grid ground truths unrecoverable. Shrink
``ALPHA_GRID`` / ``PREF_DUR_GRID_MS`` / ``SIGMA_DUR_GRID_MS`` below if the run
is too slow locally.

Usage (from project root):
    conda run --no-capture-output -n subcorticalSTRF3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_nonoise
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order as m

m.NOISE_VOXELS    = ["none"]              # noise-free: clean GT BOLD is the measurement
m.N_NOISE_REPEATS = 1                     # deterministic, repeats add nothing
m.N_VOXELS        = 20                    # independent ground-truth voxels
m.DESIGN_SEEDS    = [1000]                # one run design

# Uncomment to shrink the candidate grid for a faster (coarser) trial:
# m.ALPHA_GRID        = [1.0, 2.0, 4.0, 8.0]
# m.PREF_DUR_GRID_MS  = [34.89, 100.0, 250.0, 496.43]
# m.SIGMA_DUR_GRID_MS = [20.0, 50.0, 100.0]

m.OUTPUT_DIR = Path("kim2024_style_duration_recovery_torch_order_nonoise_results")
m.RUN_DIR    = m.OUTPUT_DIR / m.RUN_TIMESTAMP


if __name__ == "__main__":
    m.main()
