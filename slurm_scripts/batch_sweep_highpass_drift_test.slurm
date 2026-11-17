#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_highpass_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_highpass_test_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_highpass_test_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Drift-removal test (solver/sweep_highpass_drift_test.py): the same voxels and the same noise draws are
# matched run by run twice, as before ("none") and after removing the slow DCT components (period longer than
# the cutoff) from BOTH the measurement and every template ("hp128", "hp64"). The question: does the fixed
# drift of the noise model cause the run-specific favourite tau? Additive noise (jitter 0.1), clean ground
# truth, single-run estimates of all 24 run orders. Writes
#   sweep_recovery_results/<timestamp>_highpass_test_<noise>/highpass_test_<noise>.npz
#
# Needs, on this machine:
#   models_output/sweep_design_erb_3band/combined/             the 25 combined cochlear results
#   sweep_recovery_multirun_grids/run_grid_design1000..1023.npz   from batch_sweep_grid_multirun.slurm
#
# Environment variables:
#   NOISE=mid (default)   N_VOXELS=100   N_REPEATS=200   CUTOFFS="128 64"   GRID_DIR=sweep_recovery_multirun_grids
# Smoke run first (3 voxels, 5 repeats):
#   N_VOXELS=3 N_REPEATS=5 sbatch --cpus-per-task=4 --mem=16G --time=00:15:00 slurm_scripts/batch_sweep_highpass_drift_test.slurm
# Full run:
#   sbatch slurm_scripts/batch_sweep_highpass_drift_test.slurm
#
# --time=00:30:00 is only a ceiling (a local estimate for the full run is a few minutes on 14 cores);
# check `seff <jobid>` and adjust.

NOISE=${NOISE:-mid}
N_VOXELS=${N_VOXELS:-100}
N_REPEATS=${N_REPEATS:-200}
CUTOFFS=${CUTOFFS:-"128 64"}
GRID_DIR=${GRID_DIR:-sweep_recovery_multirun_grids}

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "noise=$NOISE n_voxels=$N_VOXELS n_repeats=$N_REPEATS cutoffs=$CUTOFFS grid_dir=$GRID_DIR"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_highpass_drift_test \
    --results-dir models_output/sweep_design_erb_3band/combined \
    --grid-dir "$GRID_DIR" \
    --output-dir sweep_recovery_results \
    --noise $NOISE \
    --n-voxels $N_VOXELS \
    --n-repeats $N_REPEATS \
    --cutoffs $CUTOFFS \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
