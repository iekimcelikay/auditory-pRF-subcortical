#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_recovery_gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_gt_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_gt_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Script 2 of the averaged-noisy ground truth (solver/sweep_recovery_from_groundtruth.py): recovery of
# (cf_index, alpha, tau_ms, w) from the AVERAGED ground-truth signal of
# sweep_groundtruth_distribution.py plus fresh noise (same PmNoise, jitter 0.1, separate seed stream),
# from 24 runs: joint fit and random effects (log-space average) and the spread of the 24 per-run
# estimates. Each noise level is recovered against its own averaged ground truth. Voxels are spread
# over all CPUs of the job.
#
# Needs, on this machine:
#   GROUNDTRUTH_DIR   folder written by batch_sweep_groundtruth_distribution.slurm, with
#                     groundtruth_mean_<low|mid|high>.npz (the 4 GB distribution files are NOT read)
#   sweep_recovery_multirun_grids/run_grid_design1000..1023.npz   from batch_sweep_grid_multirun.slurm
#
# Environment variables (GROUNDTRUTH_DIR is required):
#   GROUNDTRUTH_DIR=<folder>   NOISE="low mid high" (default)   N_REPEATS=1000   GRID_DIR=sweep_recovery_multirun_grids
# Smoke run first (5 repeats; read the time in the .err log):
#   GROUNDTRUTH_DIR=<folder> NOISE=low N_REPEATS=5 sbatch slurm_scripts/batch_sweep_recovery_from_groundtruth.slurm
#   (use the smoke ground truth of N_VOXELS=3 N_REPEATS=5 from the distribution job)
# Full run:
#   GROUNDTRUTH_DIR=<folder> sbatch slurm_scripts/batch_sweep_recovery_from_groundtruth.slurm
#
# --time=01:00:00 is only a ceiling (the 24-run recovery took about the same work per noise level);
# check `seff <jobid>` after the smoke run and adjust.

NOISE=${NOISE:-"low mid high"}
N_REPEATS=${N_REPEATS:-1000}
GRID_DIR=${GRID_DIR:-sweep_recovery_multirun_grids}
if [ -z "$GROUNDTRUTH_DIR" ]; then
    echo "GROUNDTRUTH_DIR is not set (folder with groundtruth_mean_<noise>.npz from the distribution job)" >&2
    exit 2
fi

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "noise=$NOISE n_repeats=$N_REPEATS groundtruth=$GROUNDTRUTH_DIR grid_dir=$GRID_DIR"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_recovery_from_groundtruth \
    --groundtruth-dir "$GROUNDTRUTH_DIR" \
    --grid-dir "$GRID_DIR" \
    --output-dir sweep_recovery_results \
    --noise $NOISE \
    --n-repeats $N_REPEATS \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
