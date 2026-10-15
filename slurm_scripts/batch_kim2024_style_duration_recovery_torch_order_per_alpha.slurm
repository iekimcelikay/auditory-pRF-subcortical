#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_duration_recovery_torch_order_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Duration-only recovery test using torch_pipeline.py's stage order + fixed
# per-CF reference calibration -- see
# kim2024_style_duration_recovery_torch_order.py's module docstring.
#
# SLURM array: one design seed per task (--array=0-9, see the script's
# SLURM_ARRAY_TASK_ID handling). Job 7686557 (8 CPUs) hit the 12 h limit
# (MaxRSS 8.5 GB, ~1 GB per worker) at 43% CPU utilisation; recovery is now
# voxel-parallel and chunked over repeats. Measured on array task 7692950_0
# (16 CPUs): 27m29s elapsed, 81% CPU efficiency, MaxRSS 4.2 GB. So --mem=16G
# (~4x headroom) and --time=01:30:00 (~3x). Hyperion CPU nodes have 48 cores /
# 94-257 GB, so 16G lets three 16-CPU tasks share a node.

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --no-capture-output matters for live progress monitoring -- see
# batch_kim2024_style_recovery_torch_order_per_alpha.slurm's comment.
#
# NOTE: if solver/ was added to this checkout after strf3.9's last editable
# install, rerun: conda run -n strf3.9 pip install -e .
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
