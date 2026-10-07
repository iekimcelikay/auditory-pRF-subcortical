#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=grid_recovery_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/grid_recovery_per_alpha_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/grid_recovery_per_alpha_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Per-alpha-pool variant of batch_grid_recovery.slurm: opens a fresh worker
# pool for each alpha value instead of one pool for the whole grid, bounding
# each worker's memory to one alpha's sharpened data (~120MB) instead of up
# to all 7 (~840MB) -- see grid_recovery_parallel_per_alpha.py's module
# docstring. Meant for fixed-memory nodes (no option to request more RAM).
#
# --cpus-per-task controls the candidate-grid build's parallelism (see
# grid_recovery_parallel_per_alpha.py: N_WORKERS defaults to $SLURM_CPUS_PER_TASK).

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
conda run -n strf3.9 python -m auditory_prf.prf_pipeline.solver.grid_recovery_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
