#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_duration_recovery_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_per_alpha_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_per_alpha_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Kim et al. 2024-style duration-model-only (AdapTrans off) recovery test
# (independent random voxels, Monte Carlo noise repeats per voxel, post-hoc
# Pearson r accuracy reporting -- see kim2024_style_duration_recovery.py's
# module docstring), per-alpha-pool parallel candidate-grid build -- see
# batch_grid_recovery_per_alpha.slurm for the memory-bounding rationale
# (fixed 16-core/32GB dIPC nodes).
#
# --cpus-per-task controls the candidate-grid build's parallelism (see
# kim2024_style_duration_recovery_parallel_per_alpha.py /
# duration_recovery_parallel_per_alpha.py: N_WORKERS defaults to
# $SLURM_CPUS_PER_TASK).
#
# Lowered from 16 to 8 -- a real run (job 7674151) silently stalled on the
# 7th/last alpha pool with zero surviving worker processes (likely OOM-killed
# -- this worker does no per-candidate caching, so peak memory scales
# directly with concurrent worker count, unlike the AdapTrans version).
# multiprocessing.Pool hangs forever rather than raising when workers die
# unexpectedly, so the job showed as RUNNING in squeue with no progress.
# Fewer concurrent workers reduces peak memory pressure; since there's no
# per-worker caching benefit to lose here, this costs roughly 2x wall-clock
# for the grid build, not a steep tradeoff.

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
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
