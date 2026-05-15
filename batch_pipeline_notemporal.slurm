#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=prf_notemporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-29                # one task per CF (num_cf=30)
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/prf_notemporal_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/prf_notemporal_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Running on: $SLURM_NODELIST"
echo "==================================="

module load Miniforge3
# try again
# Prevent NumPy/OpenBLAS from spawning its own threads inside each worker process.
# Without this, 8 Python workers × N numpy threads = massive oversubscription.
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

RESULTS_DIR=$AUDITORY_PRF_ROOT/models_output/dipc_test_280514_01
OUTPUT_DIR=$AUDITORY_PRF_ROOT/models_output/prf_notemporal

conda run -n strf3.9 python \
    $AUDITORY_PRF_ROOT/run_pipeline_notemporal.py \
    --cf $SLURM_ARRAY_TASK_ID \
    --results-dir $RESULTS_DIR \
    --output-dir $OUTPUT_DIR \
    --n-workers $SLURM_CPUS_PER_TASK

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
