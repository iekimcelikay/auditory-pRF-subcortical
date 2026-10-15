#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_recovery_torch_order_combined_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_combined_per_alpha_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_combined_per_alpha_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Combined AdapTrans + duration-selectivity recovery test, torch order +
# fixed per-CF reference -- see kim2024_style_recovery_torch_order_combined.py's
# module docstring.
#
# SLURM array: one design seed per task (--array=0-9, see the script's
# SLURM_ARRAY_TASK_ID handling). Job 7687058 (8 CPUs, unbounded per_seq cache)
# was OOM-killed at 33.5 GB MaxRSS after 3h42m; the cache is now single-entry
# and recovery is chunked over repeats, so 16 CPUs / 48 GB (~3 GB per worker)
# should have headroom. Hyperion CPU nodes have 48 cores / 94-257 GB; QoS
# regular allows 24 h. Lower --time once a task's real elapsed time is known.

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
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
