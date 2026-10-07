#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=grid_recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/grid_recovery_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/grid_recovery_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# --cpus-per-task controls the candidate-grid build's parallelism (see
# grid_recovery_parallel_workers.py: N_WORKERS defaults to $SLURM_CPUS_PER_TASK).
# --mem/--time are rough starting points — adjust based on how the first run
# actually scales (candidate-grid build time is reported incrementally in the
# job log as blocks of candidates complete).

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

conda run -n strf3.9 python -m auditory_prf.prf_pipeline.solver.grid_recovery_parallel_workers

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
