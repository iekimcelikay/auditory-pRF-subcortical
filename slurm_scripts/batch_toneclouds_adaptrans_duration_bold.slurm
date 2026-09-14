#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=toneclouds_adaptrans_duration_bold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-29
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/toneclouds_adaptrans_duration_bold_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/toneclouds_adaptrans_duration_bold_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on cluster: $SLURM_CLUSTER_NAME"
echo "Running on host: $SLURM_NODELIST"
echo "Job name: $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMITDIR"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

# Load required modules
module load Miniforge3

# ---- Environment Setup -------------------------------------------------------
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
RESULTS_DIR=$AUDITORY_PRF_ROOT/models_output/toneclouds_gaussianprf_20260619_0114
OUTPUT_DIR=$RESULTS_DIR/adaptrans_duration_bold_symmetric_$(date +%Y%m%d)_job${SLURM_ARRAY_JOB_ID}

# Run the Python script
# One job per CF. tau_ms is a free parameter (no CF-tau relationship).
# No --tau_ms_off passed below, so it defaults to None -> AdapTrans uses
# tau_ms for both ON and OFF (symmetric case).
# Duration Gaussian filter is enabled via --pref_dur/--sigma_dur (both in ms).
# Chunk margin (30 ms) is not a CLI flag -- it's baked into
# CHUNK_MARGIN_MS in full_pipeline_toneclouds_adaptrans.py and applied
# to every chunk_from_id() call automatically.
cd $AUDITORY_PRF_ROOT
conda run -n strf3.9 python -c "import sys; print(sys.executable)"
conda run -n strf3.9 python -m auditory_prf.prf_pipeline.full_pipeline_toneclouds_adaptrans \
    --cf $SLURM_ARRAY_TASK_ID \
    --results-dir $RESULTS_DIR \
    --output-dir $OUTPUT_DIR \
    --alpha 4 \
    --tau_ms 100 \
    --pref_dur 75 \
    --sigma_dur 50 \
    --noise_voxels low mid high \
    --noise_seed $((100 + SLURM_ARRAY_TASK_ID))

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
