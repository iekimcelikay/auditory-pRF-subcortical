#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=plot_cochleagrams
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/plot_cochleagrams_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/plot_cochleagrams_%j.err

# Cochleagrams (CF x time PNGs, overview, mean rate per CF, summary CSV) of one harmonic-sweep design, from its combined
# cochlea outputs. Needs models_output/$DESIGN/combined/*.npz, i.e. the cochlea array job and combine_cf_batches.py must
# have finished:
#   python examples/combine_cf_batches.py --run-dir models_output/$DESIGN --expected-n-batches 6
# Writes models_output/$DESIGN/cochleagrams/; every title carries the design label, e.g.
# 'Harmonic sweeps: 3 rates x 2 tones x 4 ISIs'.
#
# Usage: sbatch slurm_scripts/batch_plot_harmonic_sweep_cochleagrams.slurm
#   other design / options:  DESIGN=harmonic_sweep_armB WINDOW_S=0.7 sbatch <this script>

DESIGN=${DESIGN:-harmonic_sweep_armA}   # harmonic_sweep_armA | harmonic_sweep_armB
WINDOW_S=${WINDOW_S:-1.5}               # seconds of each trial shown
SMOOTH_MS=${SMOOTH_MS:-2.0}             # moving average along time

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "Design: $DESIGN  window: $WINDOW_S s  smooth: $SMOOTH_MS ms"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

conda run -n strf3.9 python $AUDITORY_PRF_ROOT/examples/plot_harmonic_sweep_cochleagrams.py \
    --design $DESIGN \
    --window-s $WINDOW_S \
    --smooth-ms $SMOOTH_MS
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
