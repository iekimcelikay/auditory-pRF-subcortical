#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_recovery_multirun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_multirun_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_recovery_multirun_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# 24-run recovery of (cf_index, alpha, tau_ms, w) (solver/sweep_recovery_multirun.py): per
# repeat, 24 runs with different stimulus orders and independent noise; one estimate, fixed
# effects (every run scored on its own, the r^2 tables averaged over runs, the highest mean r^2
# wins; the per-level best scores, top candidates and full scores of the first repeats are
# saved); compared with the single-run results (baseline) and with the spread of the 24
# per-run estimates. Voxels are spread over all CPUs of the job.
#
# Needs, on this machine:
#   models_output/$DESIGN/combined/        the 25 combined cochlear results
#   sweep_recovery_multirun_grids/$DESIGN/run_grid_design1000..1023.npz   from batch_sweep_grid_multirun.slurm
#   BASELINE_DIR   folder with the single-run results
#                  sweep_recovery_results_design1000_<low|mid|high>_<timestamp>.npz, e.g.
#                  sweep_recovery_results/$DESIGN/<timestamp>_low-mid-high_on_grid_design1000
#
# DESIGN (environment variable, default sweep_design_erb3band) picks the sweep design; everything of a design lives
# under its name (stimuli/sweep_design_names.py): models_output/<DESIGN>/combined, sweep_recovery_multirun_grids/<DESIGN>,
# sweep_recovery_results/<DESIGN>, sweep_groundtruth_results/<DESIGN>. e.g.  DESIGN=harmonic_sweep_armA sbatch <this script>
# Environment variables (BASELINE_DIR is required):
#   BASELINE_DIR=<folder>   NOISE="low mid high" (default)   N_VOXELS=100   N_REPEATS=1000
# NOISE_MODEL=additive|legacy (default additive): MATLAB-style noise added as is to the signal scaled by ONE
# global best-case constant (see auditory-prf-docs-all/additive_noise_design.md). 'legacy' = the earlier
# per-voxel scaling. All results made before the additive noise used 'legacy' and have to be re-run.
# The single-run baseline in BASELINE_DIR has to be made with the same NOISE_MODEL (run
# batch_sweep_recovery.slurm first); the log warns if it was not.
# HIGHPASS_CUTOFFS="128" (seconds; empty string = unfiltered only): after the unfiltered recovery, the same
# recovery is repeated once per cutoff on the SAME noise seeds, after removing the slow DCT components from the
# measurement and the templates of every run (solver/highpass.py, the SPM-style high-pass). Each filtered pass
# writes its own result folder with the suffix _hp<cutoff>; its "1 run" column is run 1 alone of that pass
# (the single-run baseline files are unfiltered). The unfiltered folder is identical to what this script wrote
# before the filter existed.
# Smoke run first (3 voxels, 5 repeats; read the per-voxel time in the log):
#   BASELINE_DIR=<folder> NOISE=low N_VOXELS=3 N_REPEATS=5 sbatch slurm_scripts/batch_sweep_recovery_multirun.slurm
# Full run:
#   BASELINE_DIR=<folder> sbatch slurm_scripts/batch_sweep_recovery_multirun.slurm
# Noise-free test (no noise, so one repeat is the whole experiment; no baseline folder needed;
# small request so that it does not wait in the queue):
#   NOISE=none N_REPEATS=1 sbatch --cpus-per-task=16 --mem=32G --time=00:30:00 \
#       slurm_scripts/batch_sweep_recovery_multirun.slurm
#
# --time=02:00:00 is only a ceiling. Expected run time roughly 10-15 min for the full run
# (that estimate was made before the switch to fixed effects: about 2.3 core-hours for the
# scoring and about as much for the per-run matching, over 32 CPUs); to be re-measured in the smoke
# run, then check `seff <jobid>` and adjust. Every cutoff adds roughly one more unfiltered-sized pass, so the
# ceiling is 02:00:00 for the default single cutoff.

DESIGN=${DESIGN:-sweep_design_erb3band}   # sweep_design_erb3band | harmonic_sweep_armA | harmonic_sweep_armB
NOISE=${NOISE:-"low mid high"}
NOISE_MODEL=${NOISE_MODEL:-additive}
N_VOXELS=${N_VOXELS:-100}
N_REPEATS=${N_REPEATS:-1000}
HIGHPASS_CUTOFFS=${HIGHPASS_CUTOFFS-"128"}
if [ -z "$BASELINE_DIR" ] && [ "$NOISE" != "none" ]; then
    echo "BASELINE_DIR is not set (folder with the single-run sweep_recovery_results_design1000_*.npz;" \
         "only NOISE=none can run without it)" >&2
    exit 2
fi
BASELINE_ARGS=()
if [ -n "$BASELINE_DIR" ]; then
    BASELINE_ARGS=(--baseline-dir "$BASELINE_DIR")
fi

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "design=$DESIGN noise=$NOISE noise_model=$NOISE_MODEL n_voxels=$N_VOXELS n_repeats=$N_REPEATS baseline=$BASELINE_DIR highpass=[$HIGHPASS_CUTOFFS]"
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
    --design $DESIGN \
    "${BASELINE_ARGS[@]}" \
    --noise $NOISE \
    --noise-model $NOISE_MODEL \
    --n-voxels $N_VOXELS \
    --n-repeats $N_REPEATS \
    --highpass-cutoffs $HIGHPASS_CUTOFFS \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
