#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=cochlea_sweep_cfb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --array=0-143
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweepcfb_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweepcfb_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Full spike-train cochlea model for centered-sweep WAVs, split over CF batches.
# One task = one WAV x one CF batch. With the settings below:
#   30 CFs / 5 per batch = 6 batches, 24 WAVs  ->  24 * 6 = 144 tasks (--array=0-143).
#   wav_index = task / N_BATCHES,  cf_batch_index = task % N_BATCHES
# If you change N_WAV, N_CF or CF_BATCH_SIZE, change --array to 0-(N_WAV*N_BATCHES-1).
# --mem=100G is a first guess (the unbatched jobs used 350G for all 30 CFs); check
# the real peak with `seff <jobid>` after the first tasks finish and adjust.
#
# Afterwards, recombine the batches (once all tasks are done):
#   python examples/combine_cf_batches.py --run-dir models_output/<wav_dir_name> \
#       --expected-n-batches 6
#
# Usage: WAV_DIR=/path/to/sweep_wavs sbatch slurm_scripts/batch_cochlea_sweep_cfbatch.slurm

N_WAV=24
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
# Folder holding the sweep WAVs; no default on purpose (pass it with WAV_DIR=...)
WAV_DIR=${WAV_DIR:?set WAV_DIR to the folder of sweep WAVs}

cd $AUDITORY_PRF_ROOT
conda run -n strf3.9 python $AUDITORY_PRF_ROOT/examples/cochlea_sweep_wavs.py \
    --wav-dir $WAV_DIR \
    --wav-index $WAV_INDEX \
    --cf-batch-size $CF_BATCH_SIZE \
    --cf-batch-index $CF_BATCH_INDEX

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
