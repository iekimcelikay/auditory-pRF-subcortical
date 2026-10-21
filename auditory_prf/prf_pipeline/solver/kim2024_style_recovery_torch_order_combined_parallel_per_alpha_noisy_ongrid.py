"""kim2024_style_recovery_torch_order_combined_parallel_per_alpha_noisy_ongrid.py
================================================================================
ON-GRID ground truths with the noise sweep for the combined AdapTrans +
duration-selectivity recovery
(``kim2024_style_recovery_torch_order_combined_parallel_per_alpha.py``). Does
NOT modify that script's logic -- overrides module globals before calling its
``main()``.

Changes relative to the noisy (off-grid) script:
  - ``GT_MODE = "on_grid"``: ground truths are random candidate-grid rows, so
    recovery error is due to noise / identifiability, not grid quantization.
  - PmNoise jitter is exposed as ``NOISE_JITTER`` (frequency, amplitude, phase
    SD as a fraction of nominal), [0, 0, 0] for the first run, same as the
    duration noisy on-grid runner.
Noise levels (low/mid/high), ``N_NOISE_REPEATS``, ``N_VOXELS`` and the design
seeds (via ``SLURM_ARRAY_TASK_ID``) are inherited unchanged. No ``"none"`` row.
Results go to their own folder.

Usage (from project root, via SLURM -- see
slurm_scripts/batch_kim2024_style_recovery_torch_order_combined_per_alpha_noisy_ongrid.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined_parallel_per_alpha_noisy_ongrid
"""

from functools import partial
from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined as b
import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined_parallel_per_alpha as p

# PmNoise jitter [frequency, amplitude, phase] as fraction of nominal.
NOISE_JITTER = [0.0, 0.0, 0.0]

p.GT_MODE = "on_grid"
# draw_noisy_measurement was imported BY VALUE into ``p``; wrap it there so every
# worker (forked after this patch) draws jittered noise.
p.draw_noisy_measurement = partial(p.draw_noisy_measurement, jitter=NOISE_JITTER)

OUTPUT_DIR = Path("kim2024_style_recovery_torch_order_combined_noisy_ongrid_results")
b.OUTPUT_DIR = OUTPUT_DIR
b.RUN_DIR    = OUTPUT_DIR / b.RUN_TIMESTAMP
p.RUN_DIR    = b.RUN_DIR


if __name__ == "__main__":
    print(f"[combined noisy_ongrid] GT_MODE={p.GT_MODE}, noise levels={p.NOISE_VOXELS}, "
          f"N_VOXELS={p.N_VOXELS}, N_NOISE_REPEATS={p.N_NOISE_REPEATS:,}, jitter={NOISE_JITTER}")
    p.main()
