#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=duration_recovery_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/duration_recovery_per_alpha_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/duration_recovery_per_alpha_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Duration-model-only (AdapTrans off) recovery test, per-alpha-pool parallel
# -- see batch_grid_recovery_per_alpha.slurm for the memory-bounding
# rationale (fixed 16-core/32GB nodes). No per-worker caching benefit here
# (per_seq depends on the duration params too, see
# duration_recovery_parallel_per_alpha.py's module docstring), so expect a
# flatter, more linear-in-candidate-count runtime than the AdapTrans version.

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

# NOTE: if solver/ was added to this checkout after strf3.9's last editable
# install, rerun: conda run -n strf3.9 pip install -e .
# --no-capture-output matters for live progress monitoring: plain
# `conda run` withholds the child process's stdout until it exits, which
# would otherwise make this job's .out log look frozen for the whole
# candidate-grid build even though it's actively running -- per project
# discussion.
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.duration_recovery_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
