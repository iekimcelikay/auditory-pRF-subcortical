#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=tau_sus_rec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/tau_sus_rec_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/tau_sus_rec_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=i.ekimcelikay@gmail.com

echo "Task $SLURM_ARRAY_TASK_ID / 19 started at $(date) on $SLURM_NODELIST"

module load Miniforge3

AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

conda run -n strf3.9 python tau_sus_recovery_worker.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --out_dir $AUDITORY_PRF_ROOT/tau_sus_recovery_results

echo "Task $SLURM_ARRAY_TASK_ID done at $(date)"
