#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_duration_recovery_torch_order_per_alpha_nonoise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_nonoise_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_nonoise_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# NOISE-FREE duration-only recovery test (torch order, population-pool
# normalization) -- see
# kim2024_style_duration_recovery_torch_order_parallel_per_alpha_nonoise.py.
#
# SLURM array: one design seed per task (--array=0-9, see the script's
# SLURM_ARRAY_TASK_ID handling), noise level 'none', 1 repeat per voxel. Cost is dominated by the parallel candidate-grid build, same as one
# array task of batch_kim2024_style_duration_recovery_torch_order_per_alpha.slurm
# (measured there: 27m29s on 16 CPUs, MaxRSS 4.2 GB), so the same --mem=16G and
# --time=01:30:00 are reused. The noise-free recovery step itself is cheap
# (100 voxels x 1 repeat). Adjust after the first run.

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
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha_nonoise

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
