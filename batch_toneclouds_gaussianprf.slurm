#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=toneclouds_gaussianprf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=350G
#SBATCH --array=0-24
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/toneclouds_gaussianprf_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/toneclouds_gaussianprf_%A_%a.err
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

# Run the Python script
cd $AUDITORY_PRF_ROOT
conda run -n strf3.9 python -c "import sys; print(sys.executable)"
conda run -n strf3.9 python $AUDITORY_PRF_ROOT/examples/cochlea_toneclouds_gaussianprf.py \
    --wav-index $SLURM_ARRAY_TASK_ID \
    --wav-dir /scratch/ecelikay/workspace/auditory-pRF-subcortical/stimuli/produced/toneclouds_gaussianprf_20260619_0114

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
