#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=sweep_noise_scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_noise_scale_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/sweep_noise_scale_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Noise-scale sweep (solver/sweep_noise_scale_recovery.py): at which noise level does the perfect recovery
# turn into the 'regression to the mean'? Ground truth = the CLEAN signal of the 100 on-grid voxels (as in the
# perfect noise-free recovery), 24 run orders; noise = k x the 'low' preset (white, cardiac, respiratory and
# drift together, jitter 0.1, added as is to the signal scaled by the global best-case constant). k = 0 is
# noise-free, k = 1 is the low noise. The noise of every level is k times the same pattern (same seeds).
# Per level: run 1 alone and the 24-run fixed-effects estimate (mean r^2 over runs) from identical noise, tau
# heatmaps and tau summary.
# Every noise draw is matched without a filter and once per high-pass cutoff (HIGHPASS_CUTOFFS, seconds; the
# DCT filter of the high-pass test, applied to the measurement and to the templates of every run), on the SAME
# noise, so the differences are due to the filter. Files of the unfiltered recovery keep their names, the others
# carry hp<cutoff>. Saved for every matching: the 10 best candidates with their Pearson r and the best r at every
# CF, alpha, tau and w level (match_* arrays in the npz files), and with SAVE_MEASUREMENTS=1 the noisy
# measurements (measurements_k<k>_*.npy, about 0.8 GB per level) so that any later analysis can be done locally.
# Writes sweep_recovery_results/<timestamp>_noise_scale_<preset>_24runs/
#
# Needs, on this machine:
#   sweep_recovery_multirun_grids/run_grid_design1000..1023.npz   from batch_sweep_grid_multirun.slurm
#   models_output/sweep_design_erb_3band/combined                 (only for the CF list)
#
# Environment variables (all optional):
#   PRESET=low   NOISE_SCALES="0 0.01 0.03 0.1 0.3 1 3"   N_REPEATS=200   GRID_DIR=sweep_recovery_multirun_grids
#   HIGHPASS_CUTOFFS="128 64" (empty string = no filter)   SAVE_MEASUREMENTS=1 (0 = do not save them)
# Smoke run first (2 voxels' worth of time: 5 repeats, 3 levels):
#   NOISE_SCALES="0 0.3 3" N_REPEATS=5 sbatch --cpus-per-task=8 --mem=8G --time=00:15:00 slurm_scripts/batch_sweep_noise_scale_recovery.slurm
# Full run:
#   sbatch slurm_scripts/batch_sweep_noise_scale_recovery.slurm
#
# --time=04:00:00 is only a ceiling (the filters triple the matching work); check `seff <jobid>` and adjust.

PRESET=${PRESET:-low}
NOISE_SCALES=${NOISE_SCALES:-"0 0.01 0.03 0.1 0.3 1 3"}
N_REPEATS=${N_REPEATS:-200}
GRID_DIR=${GRID_DIR:-sweep_recovery_multirun_grids}
HIGHPASS_CUTOFFS=${HIGHPASS_CUTOFFS-"128 64"}
SAVE_MEASUREMENTS=${SAVE_MEASUREMENTS:-1}
SAVE_FLAG=""
if [ "$SAVE_MEASUREMENTS" = "1" ]; then SAVE_FLAG="--save-measurements"; fi

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "preset=$PRESET noise_scales=$NOISE_SCALES n_repeats=$N_REPEATS grid_dir=$GRID_DIR highpass=[$HIGHPASS_CUTOFFS] save_measurements=$SAVE_MEASUREMENTS"
echo "==================================="

module load Miniforge3
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p outputs_terminal
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.sweep_noise_scale_recovery \
    --grid-dir "$GRID_DIR" \
    --output-dir sweep_recovery_results \
    --preset $PRESET \
    --noise-scales $NOISE_SCALES \
    --n-repeats $N_REPEATS \
    --highpass-cutoffs $HIGHPASS_CUTOFFS \
    $SAVE_FLAG \
    --n-workers $SLURM_CPUS_PER_TASK
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
