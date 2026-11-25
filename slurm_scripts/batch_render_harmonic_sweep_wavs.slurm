#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=render_harmonic_sweeps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/render_harmonic_sweeps_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/render_harmonic_sweeps_%j.err

# Render the harmonic-sweep WAVs (family P2, auditory-prf-docs-all/fm_sweep_silence_kim_design.md) on the cluster:
# 24 mono 16-bit trials + silence_20s.wav + design_info.json per design, in stimuli/produced/<design name>/
# (the folder is named like the design, stimuli/sweep_design_names.py):
#   harmonic_sweep_armA   3 rates x 2 tones x 4 ISIs
#   harmonic_sweep_armB   3 rates x 4 tones x 2 ISIs
# Default: both layouts (~1 min). Then run the cochlea model on each folder:
#   DESIGN=harmonic_sweep_armA sbatch slurm_scripts/batch_cochlea_sweep_cfbatch.slurm
#
# Usage: sbatch slurm_scripts/batch_render_harmonic_sweep_wavs.slurm
#   one layout only:  LAYOUT=A sbatch slurm_scripts/batch_render_harmonic_sweep_wavs.slurm     (A | B | both)

LAYOUT=${LAYOUT:-both}

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "Layout: $LAYOUT"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p $AUDITORY_PRF_ROOT/stimuli/produced

conda run -n strf3.9 python $AUDITORY_PRF_ROOT/stimuli/render_harmonic_sweep_wavs.py \
    --layout $LAYOUT \
    --out-root $AUDITORY_PRF_ROOT/stimuli/produced
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
