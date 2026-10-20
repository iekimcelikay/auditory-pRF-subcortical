"""kim2024_style_recovery_torch_order_parallel_per_alpha_noisy_ongrid.py
========================================================================
ON-GRID ground truths with the noise sweep for the AdapTrans tau recovery
(``kim2024_style_recovery_torch_order_parallel_per_alpha.py``). Does NOT modify
that script's logic -- overrides module globals before calling its ``main()``.

Changes relative to the noisy script:
  - ``GT_MODE = "on_grid"``: ground truths are random candidate-grid rows, so
    recovery error is due to noise / identifiability, not grid quantization.
Noise levels (low/mid/high), ``N_NOISE_REPEATS``, ``N_VOXELS`` and the design
seeds are inherited unchanged. No ``"none"`` row (see the nonoise runner).
Results go to their own folder.

Usage (from project root, via SLURM -- see
slurm_scripts/batch_kim2024_style_recovery_torch_order_per_alpha_noisy_ongrid.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_parallel_per_alpha_noisy_ongrid
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order as b
import auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_parallel_per_alpha as p

p.GT_MODE = "on_grid"

OUTPUT_DIR = Path("kim2024_style_recovery_torch_order_noisy_ongrid_results")
b.OUTPUT_DIR = OUTPUT_DIR
b.RUN_DIR    = OUTPUT_DIR / b.RUN_TIMESTAMP
p.RUN_DIR    = b.RUN_DIR


if __name__ == "__main__":
    print(f"[tau noisy_ongrid] GT_MODE={p.GT_MODE}, noise levels={p.NOISE_VOXELS}, "
          f"N_VOXELS={p.N_VOXELS}, N_NOISE_REPEATS={p.N_NOISE_REPEATS:,}")
    p.main()
