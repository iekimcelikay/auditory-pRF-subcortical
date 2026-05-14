#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=auditory_prf
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=64      # CPUs for parallel data generation
#SBATCH --mem-per-cpu=30G
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

# ---- Environment Setup --
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical

# Your command
# Run the Python script
cd $AUDITORY_PRF_ROOT
conda run -n strf3.9 python -c "import sys; print(sys.executable)"
conda run -n strf3.9 python /scratch/ecelikay/workspace/auditory-pRF-subcortical/examples/cochlea_customwas_example.py

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="