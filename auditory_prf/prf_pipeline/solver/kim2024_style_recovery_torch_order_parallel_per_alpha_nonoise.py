"""kim2024_style_recovery_torch_order_parallel_per_alpha_nonoise.py
===================================================================
Noise-FREE run of ``kim2024_style_recovery_torch_order_parallel_per_alpha.py``
(AdapTrans tau recovery: parallel candidate-grid build + voxel-parallel
recovery, torch order, population-pool alpha normalization). Does NOT modify
that script's logic -- overrides its module globals in place before calling its
own ``main()``.

With no noise the measurement is deterministic, so one "repeat" per voxel is
enough. With ``GT_MODE = "on_grid"`` recovery should be exact; any miss comes
from the model (identifiability), not noise or grid quantization. Switch to
``"off_grid"`` afterwards to measure quantization error. The tau grid is the
script's own (tau_on == tau_off, 20-400 ms); taus are not constrained across CFs.

Patched on the parallel module ``p``: ``NOISE_VOXELS``, ``N_NOISE_REPEATS``,
``N_VOXELS`` and ``GT_MODE`` (the first three were imported BY VALUE). The base
module ``b`` owns the output-path helpers, which read ``b.RUN_DIR``.
``DESIGN_SEEDS`` is not overridden: ``SLURM_ARRAY_TASK_ID`` picks the seed.

Usage (from project root, via SLURM -- see
slurm_scripts/batch_kim2024_style_recovery_torch_order_per_alpha_nonoise.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_parallel_per_alpha_nonoise
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order as b
import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_parallel_per_alpha as p

p.NOISE_VOXELS    = ["none"]   # noise-free: clean GT BOLD is the measurement
p.N_NOISE_REPEATS = 1          # deterministic, repeats add nothing
p.N_VOXELS        = 100        # independent ground-truth voxels (cheap: 1 repeat each)
p.GT_MODE         = "on_grid"  # step 1: ON the candidate grid; switch to "off_grid" after

OUTPUT_DIR = Path("kim2024_style_recovery_torch_order_nonoise_results")
b.OUTPUT_DIR = OUTPUT_DIR
b.RUN_DIR    = OUTPUT_DIR / b.RUN_TIMESTAMP
p.RUN_DIR    = b.RUN_DIR


if __name__ == "__main__":
    print(f"[tau nonoise] GT_MODE={p.GT_MODE}, noise levels={p.NOISE_VOXELS}, "
          f"N_VOXELS={p.N_VOXELS}, N_NOISE_REPEATS={p.N_NOISE_REPEATS}")
    p.main()
