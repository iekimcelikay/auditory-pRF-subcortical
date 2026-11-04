#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=cochlea_sweep_cfb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --array=0-149
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweepcfb_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweepcfb_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Full spike-train cochlea model for centered-sweep WAVs, split over CF batches.
# One task = one WAV x one CF batch. With the settings below:
#   30 CFs / 5 per batch = 6 batches, 25 WAVs  ->  25 * 6 = 150 tasks (--array=0-149).
#   WAV 0-23 = the sweeps (tasks 0-143); WAV 24 = silence_20s.wav (tasks 144-149),
#   e.g. only the silent trial:  sbatch --array=144-149 <this script>
#   wav_index = task / N_BATCHES,  cf_batch_index = task % N_BATCHES
# If you change N_WAV, N_CF or CF_BATCH_SIZE, change --array to 0-(N_WAV*N_BATCHES-1).
# --mem=80G: measured peak was 57.7 GB for one 5-CF task (~11.5 GB per CF; wav 7, cf
# batch 1, job 7836578_43, 6 min), plus ~40% headroom. Only one task was measured; if
# `sacct -j <jobid> --format=JobID,State,MaxRSS` shows OUT_OF_MEMORY, resubmit those
# tasks with --mem=100G.
#
# Afterwards, recombine the batches (once all tasks are done):
#   python examples/combine_cf_batches.py --run-dir models_output/<wav_dir_name> \
#       --expected-n-batches 6
#
# Usage: sbatch slurm_scripts/batch_cochlea_sweep_cfbatch.slurm
#   (WAV_DIR defaults to stimuli/produced/sweep_design_erb_3band; override with WAV_DIR=...)

N_WAV=25
N_CF=30
CF_BATCH_SIZE=5
N_BATCHES=$(( (N_CF + CF_BATCH_SIZE - 1) / CF_BATCH_SIZE ))
WAV_INDEX=$(( SLURM_ARRAY_TASK_ID / N_BATCHES ))
CF_BATCH_INDEX=$(( SLURM_ARRAY_TASK_ID % N_BATCHES ))

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID  (wav $WAV_INDEX, cf batch $CF_BATCH_INDEX of $N_BATCHES)"
echo "Running on host: $SLURM_NODELIST"
echo "Job name: $SLURM_JOB_NAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

if [ "$WAV_INDEX" -ge "$N_WAV" ]; then
    echo "wav index $WAV_INDEX >= N_WAV $N_WAV: array is larger than the design, nothing to do"
    exit 0
fi

# Load required modules
module load Miniforge3

# ---- Environment Setup --
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
# Folder holding the 24 sweep WAVs (copy sweep_design_erb_3band/ from the local machine here)
WAV_DIR=${WAV_DIR:-$AUDITORY_PRF_ROOT/stimuli/produced/sweep_design_erb_3band}

cd $AUDITORY_PRF_ROOT
# Shared parent of all tasks' output folders; mkdir -p is safe when tasks start together
mkdir -p $AUDITORY_PRF_ROOT/models_output/$(basename $WAV_DIR)

conda run -n strf3.9 python $AUDITORY_PRF_ROOT/examples/cochlea_sweep_wavs.py \
    --wav-dir $WAV_DIR \
    --wav-index $WAV_INDEX \
    --cf-batch-size $CF_BATCH_SIZE \
    --cf-batch-index $CF_BATCH_INDEX
PYTHON_STATUS=$?

# End of job
echo "==================================="
echo "Job finished at $(date) with python exit status $PYTHON_STATUS"
echo "==================================="
# Report a crash as FAILED in sacct instead of COMPLETED
exit $PYTHON_STATUS
