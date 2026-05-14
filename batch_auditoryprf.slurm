#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=auditory_prf
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=32      # CPUs for parallel data generation
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/test_auditoryprf_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/test_auditoryprf_%j.err
#SBATCH --mail-type=ALL           # Notifications for job done or fail
#SBATCH --mail-user=e.celikay@bcbl.eu # Email address for notifications

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

# Initialize conda shell functions (required for conda activate in non-interactive SLURM)
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh

# Activate conda environment
conda activate strf3.9
python -c "import sys; print(sys.executable)"

# ---- Environment Setup --
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical

# Your command
# Run the Python script
cd $AUDITORY_PRF_ROOT
python /scratch/ecelikay/workspace/auditory-pRF-subcortical/examples/cochlea_customwas_example.py

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="