#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Kim-2024-style recovery of (cf_index, alpha, tau_ms, w) for the FM-sweep forward model
# (solver/sweep_recovery.py). Grid: 30 CF x 7 alpha x 9 tau x 4 w = 7,560 templates,
# built in parallel over the 210 (alpha, CF) pairs with --cpus-per-task workers.
#
# Defaults = the first test: NOISE-FREE, ON-GRID ground truths, 100 voxels, 1 repeat,
# design seed 1000. Override with environment variables:
#   NOISE=none|low|mid|high|all   (or several: NOISE="low mid high"; 'all' = low mid high,
#   one shared grid and the same ground truths, plus the noise-comparison plots)
#   GT_MODE=on_grid|off_grid   N_VOXELS=100   N_REPEATS=<n>  (default 10,000 per noisy level)
#   GRID_FILE=<path to a saved candidate_grid_design<seed>.npz>   (skip the grid build)
# NOISE_MODEL=additive|legacy (default additive): MATLAB-style noise added as is to the signal scaled by ONE
# global best-case constant (see auditory-prf-docs-all/additive_noise_design.md). 'legacy' = the earlier
# per-voxel scaling. All results made before the additive noise used 'legacy' and have to be re-run.
# e.g.  NOISE=low N_REPEATS=10000 sbatch --time=12:00:00 slurm_scripts/batch_sweep_recovery.slurm
# Design seeds 1000...1009 as an array:  sbatch --array=0-9 slurm_scripts/batch_sweep_recovery.slurm
#
# --mem=16G / --time=02:00:00 are first guesses for the noise-free test (~250 MB per worker
# locally); check `seff <jobid>` and adjust. Noisy runs (10,000 repeats x 100 voxels) need a
# longer --time; measure the noise-free run first.
#
# DESIGN (environment variable, default sweep_design_erb3band) picks the sweep design; everything of a design lives
# under its name (stimuli/sweep_design_names.py): models_output/<DESIGN>/combined, sweep_recovery_multirun_grids/<DESIGN>,
# sweep_recovery_results/<DESIGN>, sweep_groundtruth_results/<DESIGN>. e.g.  DESIGN=harmonic_sweep_armA sbatch <this script>
# Needs the 25 combined results of the design on this machine: models_output/$DESIGN/combined/ (24 trials + silence_20s)

DESIGN=${DESIGN:-sweep_design_erb3band}   # sweep_design_erb3band | harmonic_sweep_armA | harmonic_sweep_armB
NOISE=${NOISE:-none}
NOISE_MODEL=${NOISE_MODEL:-additive}
GT_MODE=${GT_MODE:-on_grid}
N_VOXELS=${N_VOXELS:-100}
DESIGN_SEED=$(( 1000 + ${SLURM_ARRAY_TASK_ID:-0} ))
REPEATS_ARG=""
if [ -n "$N_REPEATS" ]; then REPEATS_ARG="--n-repeats $N_REPEATS"; fi
# GRID_FILE=<candidate_grid_design1000.npz> reuses a saved grid and skips the ~10 min build
# (the grid depends only on the design seed, not on noise or ground truths).
if [ -n "$GRID_FILE" ]; then REPEATS_ARG="$REPEATS_ARG --grid-file $GRID_FILE"; fi

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID  array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "design=$DESIGN noise=$NOISE noise_model=$NOISE_MODEL gt_mode=$GT_MODE n_voxels=$N_VOXELS design_seed=$DESIGN_SEED $REPEATS_ARG"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_recovery \
    --design $DESIGN \
    --noise $NOISE --noise-model $NOISE_MODEL --gt-mode $GT_MODE --n-voxels $N_VOXELS \
    --design-seed $DESIGN_SEED $REPEATS_ARG
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
