#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=cochlea_sweep_0p03oct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=350G
#SBATCH --array=0-23
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweep003_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/cochlea_sweep003_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Full spike-train cochlea model (CochleaWavSimulation) for the 24 borderline
# (0.03 oct) centered-sweep stimuli: 3 centers x 8 cycle counts, one WAV per task.
# Array index = position of the WAV in the sorted file list of WAV_DIR:
#   0-7   fc605.0hz   (2, 3, 5, 7, 11, 17, 26, 40 cycles)
#   8-15  fc774.6hz
#   16-23 fc991.7hz

# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
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
# Folder holding the 24 WAVs (copy shift_0.03oct/ from the local machine here)
WAV_DIR=$AUDITORY_PRF_ROOT/stimuli/produced/sweep_design_0p03oct

cd $AUDITORY_PRF_ROOT
conda run -n strf3.9 python -c "import sys; print(sys.executable)"
conda run -n strf3.9 python $AUDITORY_PRF_ROOT/examples/cochlea_customwas_example.py \
    --wav-dir $WAV_DIR \
    --wav-index $SLURM_ARRAY_TASK_ID

# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
