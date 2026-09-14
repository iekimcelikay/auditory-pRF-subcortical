#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=tau_rec_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/tau_rec_agg_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/tau_rec_agg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e.celikay@bcbl.eu

echo "Aggregation started at $(date)"

module load Miniforge3

AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

conda run -n strf3.9 python tau_recovery_aggregate.py \
    --results_dir $AUDITORY_PRF_ROOT/tau_recovery_results

echo "Done at $(date)"
