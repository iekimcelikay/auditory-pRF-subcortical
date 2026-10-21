#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_duration_recovery_torch_order_per_alpha_noisy_ongrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_noisy_ongrid_%A_%a.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_noisy_ongrid_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# ON-GRID duration-only recovery with the noise sweep (low/mid/high) and PmNoise
# PmNoise jitter (NOISE_JITTER in the runner; 0 for the first run), torch order, population-pool normalization -- see
# kim2024_style_duration_recovery_torch_order_parallel_per_alpha_noisy_ongrid.py.
#
# SLURM array: one design seed per task (--array=0-9). Noise levels, repeats and
# voxel count are inherited from the noisy script. No 'none' baseline here.
#
# Resources copied from batch_kim2024_style_duration_recovery_torch_order_per_alpha.slurm
# (27m29s, MaxRSS 4.2 GB with the old 352-pair grid; the 117-pair grid builds faster).
# Tighten --time / --mem after the first run.

echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

module load Miniforge3

export AUDITORY_PRF_ROOT=/scratch/ecelikay/workspace/auditory-pRF-subcortical
cd $AUDITORY_PRF_ROOT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NOTE: if solver/ was added to this checkout after strf3.9's last editable
# install, rerun: conda run -n strf3.9 pip install -e .
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha_noisy_ongrid

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
