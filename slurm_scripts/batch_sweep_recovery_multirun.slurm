#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_recovery_multirun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_multirun_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_multirun_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# 24-run recovery of (cf_index, alpha, tau_ms, w) (solver/sweep_recovery_multirun.py): per
# repeat, 24 runs with different stimulus orders and independent noise; two estimates, the
# joint fit (concatenated runs) and random effects (one estimate per run, then the average in
# log space); compared with the single-run results (baseline) and with the spread of the 24
# per-run estimates. Voxels are spread over all CPUs of the node.
#
# Needs, on this machine:
#   models_output/sweep_design_erb_3band/combined/        the 25 combined cochlear results
#   sweep_recovery_multirun_grids/run_grid_design1000..1023.npz   from batch_sweep_grid_multirun.slurm
#   BASELINE_DIR   folder with the single-run results
#                  sweep_recovery_results_design1000_<low|mid|high>_<timestamp>.npz
#
# Environment variables (BASELINE_DIR is required):
#   BASELINE_DIR=<folder>   NOISE="low mid high" (default)   N_VOXELS=100   N_REPEATS=1000
# Smoke run first (3 voxels, 5 repeats; read the per-voxel time in the log):
#   BASELINE_DIR=<folder> NOISE=low N_VOXELS=3 N_REPEATS=5 sbatch slurm_scripts/batch_sweep_recovery_multirun.slurm
# Full run:
#   BASELINE_DIR=<folder> sbatch slurm_scripts/batch_sweep_recovery_multirun.slurm
#
# --time=01:00:00 is only a ceiling. Expected run time roughly 5-10 min for the full run
# (about 2.3 core-hours of work for the joint fit and about as much for the per-run matching,
# over 48 CPUs); to be measured in the smoke run, then check `seff <jobid>` and adjust.

NOISE=${NOISE:-"low mid high"}
N_VOXELS=${N_VOXELS:-100}
N_REPEATS=${N_REPEATS:-1000}
if [ -z "$BASELINE_DIR" ]; then
    echo "BASELINE_DIR is not set (folder with the single-run sweep_recovery_results_design1000_*.npz)" >&2
    exit 2
fi

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "noise=$NOISE n_voxels=$N_VOXELS n_repeats=$N_REPEATS baseline=$BASELINE_DIR"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_recovery_multirun \
    recover \
    --results-dir models_output/sweep_design_erb_3band/combined \
    --grid-dir sweep_recovery_multirun_grids \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir sweep_recovery_results \
    --noise $NOISE \
    --n-voxels $N_VOXELS \
    --n-repeats $N_REPEATS \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
