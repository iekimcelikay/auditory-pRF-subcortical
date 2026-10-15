"""kim2024_style_duration_recovery_torch_order_parallel_per_alpha_nonoise.py
===========================================================================
Noise-FREE run of
``kim2024_style_duration_recovery_torch_order_parallel_per_alpha.py`` (parallel
candidate-grid build + voxel-parallel recovery, torch order, population-pool
normalization). Does NOT modify that script -- overrides its module globals
in place before calling its own (otherwise unmodified) ``main()``.

With no noise the measurement is deterministic, so one "repeat" per voxel is
enough. Recovery should match the ground truth up to grid resolution; any
remaining bias comes from the model itself (normalization, silence handling,
off-grid ground truths), not from noise.

Two modules must be patched, not one:
  - the parallel module (``p``): ``NOISE_VOXELS``, ``N_NOISE_REPEATS``,
    ``N_VOXELS``, ``DESIGN_SEEDS`` / ``N_DESIGN_SEEDS`` were imported BY VALUE
    into it, so they are read from ``p``'s own globals.
  - the base torch-order module (``b``): the output-path helpers
    (``candidate_grid_path``, ``recovery_output_paths``, ...) read ``b.RUN_DIR``.
Worker processes are forked after the patch, so they see the patched values.

Usage (from project root, request multiple CPUs via SLURM -- see
slurm_scripts/batch_kim2024_style_duration_recovery_torch_order_per_alpha_nonoise.slurm):
    conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha_nonoise
"""

from pathlib import Path

import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order as b
import auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha as p

NOISE_VOXELS_NONOISE    = ["none"]    # noise-free: clean GT BOLD is the measurement
N_NOISE_REPEATS_NONOISE = 1           # deterministic, repeats add nothing
N_VOXELS_NONOISE        = 100         # independent ground-truth voxels (cheap: 1 repeat each)
DESIGN_SEEDS_NONOISE    = [1000]      # one run design

p.NOISE_VOXELS    = NOISE_VOXELS_NONOISE
p.N_NOISE_REPEATS = N_NOISE_REPEATS_NONOISE
p.N_VOXELS        = N_VOXELS_NONOISE
p.DESIGN_SEEDS    = DESIGN_SEEDS_NONOISE
p.N_DESIGN_SEEDS  = len(DESIGN_SEEDS_NONOISE)

OUTPUT_DIR = Path("kim2024_style_duration_recovery_torch_order_nonoise_results")
b.OUTPUT_DIR = OUTPUT_DIR
b.RUN_DIR    = OUTPUT_DIR / b.RUN_TIMESTAMP
p.RUN_DIR    = b.RUN_DIR


if __name__ == "__main__":
    p.main()
