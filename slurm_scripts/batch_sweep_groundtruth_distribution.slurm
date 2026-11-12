#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_groundtruth_dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_groundtruth_dist_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_groundtruth_dist_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Script 1 of the averaged-noisy ground truth (solver/sweep_groundtruth_distribution.py): for every
# voxel, run order (24) and noise level, N_REPEATS noisy copies of the clean BOLD (PmNoise with
# jitter 0.1 on frequency, amplitude and phase). Saves, per noise level, the full distribution
#   groundtruth_distribution_<noise>.npy   (n_voxels, n_repeats, 24 runs, 408 TRs) float32
# and its mean over the repeats, the new ground truth (written last, so it marks a complete level)
#   groundtruth_mean_<noise>.npz
# into sweep_groundtruth_results/<timestamp>_groundtruth_24runs/.
#
# Disk: 3.9 GB per noise level for the full run (100 voxels x 1000 repeats), about 12 GB for all three.
#
# Needs, on this machine:
#   models_output/sweep_design_erb_3band/combined/        the 25 combined cochlear results
#   sweep_recovery_multirun_grids/run_grid_design1000.npz  (only its candidate parameters are read,
#                                                           to draw the same 100 on-grid voxels as before)
#
# Environment variables:
#   NOISE="low mid high" (default)   N_VOXELS=100   N_REPEATS=1000   GRID_DIR=sweep_recovery_multirun_grids
# Smoke run first (3 voxels, 5 repeats; read the per-voxel time in the .err log):
#   NOISE=low N_VOXELS=3 N_REPEATS=5 sbatch slurm_scripts/batch_sweep_groundtruth_distribution.slurm
# Full run:
#   sbatch slurm_scripts/batch_sweep_groundtruth_distribution.slurm
#
# --time=01:00:00 is only a ceiling; measure in the smoke run, then check `seff <jobid>` and adjust.

NOISE=${NOISE:-"low mid high"}
N_VOXELS=${N_VOXELS:-100}
N_REPEATS=${N_REPEATS:-1000}
GRID_DIR=${GRID_DIR:-sweep_recovery_multirun_grids}

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "noise=$NOISE n_voxels=$N_VOXELS n_repeats=$N_REPEATS grid_dir=$GRID_DIR"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_groundtruth_distribution \
    --results-dir models_output/sweep_design_erb_3band/combined \
    --grid-dir "$GRID_DIR" \
    --output-dir sweep_groundtruth_results \
    --noise $NOISE \
    --n-voxels $N_VOXELS \
    --n-repeats $N_REPEATS \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
