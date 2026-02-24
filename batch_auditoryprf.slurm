#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=auditory_prf
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=16      # CPUs for parallel data generation
#SBATCH --mem=256G
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/test_auditoryprf_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/test_auditoryprf_%j.err
#SBATCH --mail-type=ALL           # Notifications for job done or fail
#SBATCH --mail-user=e.celikay@bcbl.eu # Email address for notifications

# Activate conda environment
conda strf3.9

# ---- Environment Setup --
export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical

# Run the Python script
cd $AUDITORY_PRF_ROOT
python /scratch/ecelikay/workspace/auditory-pRF-subcortical/examples/cochlea_customwas_example.py

