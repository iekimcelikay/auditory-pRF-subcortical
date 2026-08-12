#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=gt_multitau_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-29
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/gt_multitau_ft_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/gt_multitau_ft_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=i.ekimcelikay@gmail.com

echo "Task $SLURM_ARRAY_TASK_ID / 29 started at $(date) on $SLURM_NODELIST"

module load Miniforge3

AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

conda run -n strf3.9 python gt_multitau_fulltimecourse_singlerun_worker.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --out_dir $AUDITORY_PRF_ROOT/gt_multitau_fulltimecourse_singlerun_results

echo "Task $SLURM_ARRAY_TASK_ID done at $(date)"
