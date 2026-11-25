#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=render_sweep_silence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/render_sweep_silence_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/render_sweep_silence_%j.err

# Render the tone/ISI FM-sweep WAVs (family P2, auditory-prf-docs-all/fm_sweep_silence_kim_design.md) on the cluster:
# 24 mono 16-bit trials per (layout, carrier) plus silence_20s.wav, one folder each under stimuli/produced/.
# Defaults below: both layouts (A: 2 tones x 4 ISIs, B: 4 tones x 2 ISIs), harmonic carrier -> 2 folders, 50 WAVs, ~1 min.
#
# Then run the cochlea model on each folder (WAV_DIR=<folder> sbatch slurm_scripts/batch_cochlea_sweep_cfbatch.slurm):
#   stimuli/produced/sweeps_silence_P2_armA_harmonic_cal60_<TAG>
#   stimuli/produced/sweeps_silence_P2_armB_harmonic_cal60_<TAG>
#
# Usage: sbatch slurm_scripts/batch_render_sweep_silence_wavs.slurm
#   override with environment variables, e.g.  LAYOUT=A CARRIER=both TAG=20261125 sbatch <this script>

LAYOUT=${LAYOUT:-both}        # A | B | both
CARRIER=${CARRIER:-harmonic}  # harmonic | sine | both
TAG=${TAG:-20261125}          # suffix of the folder names (default: the design date)

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "Layout: $LAYOUT  carrier: $CARRIER  tag: $TAG"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT
mkdir -p $AUDITORY_PRF_ROOT/stimuli/produced

conda run -n strf3.9 python $AUDITORY_PRF_ROOT/stimuli/render_sweep_silence_wavs.py \
    --layout $LAYOUT \
    --carrier $CARRIER \
    --tag $TAG \
    --out-root $AUDITORY_PRF_ROOT/stimuli/produced
PYTHON_STATUS=$?

echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
exit $PYTHON_STATUS
