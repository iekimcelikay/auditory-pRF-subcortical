#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_grid_multirun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_grid_multirun_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_grid_multirun_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Noise-free candidate grid of ONE run order per array task (24 run orders, design seeds
# 1000 ... 1023), for the 24-run recovery (solver/sweep_recovery_multirun.py).
# Each grid: 30 CF x 7 alpha x 9 tau x 4 w = 7,560 templates of one 408-TR run, saved
# UN-normalised (each run keeps its amplitude) as
#   sweep_recovery_multirun_grids/run_grid_design<seed>.npz
# The build is parallel over the 210 (alpha, CF) pairs; about 10 min with 16 CPUs (the time
# limit is only a ceiling; check `seff <jobid>` and adjust). 16 CPUs start sooner in the queue
# than a whole 48-core node; the 24 tasks run side by side.
#
# Submit:  sbatch slurm_scripts/batch_sweep_grid_multirun.slurm      (24 array tasks)
# Check:   ls sweep_recovery_multirun_grids/   -> 24 files, run_grid_design1000 ... 1023
#
# Needs the 25 combined cochlear results on this machine:
#   models_output/sweep_design_erb_3band/combined/   (24 sweeps + silence_20s)

DESIGN_SEED=$(( 1000 + ${SLURM_ARRAY_TASK_ID:-0} ))

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID  array task: ${SLURM_ARRAY_TASK_ID:-none}  design seed: $DESIGN_SEED"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal sweep_recovery_multirun_grids
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_recovery_multirun \
    build-grid \
    --design-seed $DESIGN_SEED \
    --results-dir models_output/sweep_design_erb_3band/combined \
    --output-dir sweep_recovery_multirun_grids \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
