#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_recovery_torch_order_per_alpha_noisy_ongrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_per_alpha_noisy_ongrid_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_per_alpha_noisy_ongrid_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# ON-GRID tau recovery with the noise sweep (low/mid/high); settings otherwise inherited from the noisy script.
# Resources copied from the noisy tau job below (10m04s, MaxRSS 2.7 GB).
#
# AdapTrans recovery test using torch_pipeline.py's stage order (CF-select
# before power-law sharpening) + fixed per-CF reference calibration -- see
# kim2024_style_recovery_torch_order.py's module docstring. Per-worker
# caching is still valid here (per_seq depends on cf_index+alpha only, not
# tau/w), same as the original (non-torch-order) AdapTrans parallel worker,
# so this uses the full 16 cores like that one does.
#
# --cpus-per-task controls the candidate-grid build's parallelism (see
# kim2024_style_recovery_torch_order_parallel_per_alpha.py: N_WORKERS
# defaults to $SLURM_CPUS_PER_TASK) and the voxel-parallel recovery stage.
#
# SLURM array: one design seed per task (--array=0-9, via SLURM_ARRAY_TASK_ID).
# Measured on array task 7693277_0 (16 CPUs): 10m04s elapsed, 61% CPU
# efficiency, MaxRSS 2.7 GB. So --mem=16G (~6x headroom) and --time=00:45:00
# (~4.5x). Hyperion CPU nodes have 48 cores / 94-257 GB, so 16G lets three
# 16-CPU tasks share a node.

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

# Prevent NumPy/OpenBLAS from spawning its own threads inside each worker
# process -- with 16 worker processes already saturating the node's 16
# cores, un-pinned BLAS threading would oversubscribe them.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --no-capture-output matters for live progress monitoring: plain
# `conda run` withholds the child process's stdout until it exits, which
# would otherwise make this job's .out log look frozen for the whole
# candidate-grid build even though it's actively running.
#
# NOTE: if solver/ was added to this checkout after strf3.9's last editable
# install, rerun: conda run -n strf3.9 pip install -e .
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_parallel_per_alpha_noisy_ongrid

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
