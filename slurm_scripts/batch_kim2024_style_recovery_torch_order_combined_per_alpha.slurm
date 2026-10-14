#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_recovery_torch_order_combined_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_combined_per_alpha_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_recovery_torch_order_combined_per_alpha_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Combined AdapTrans + duration-selectivity recovery test, torch order +
# fixed per-CF reference -- see kim2024_style_recovery_torch_order_combined.py's
# module docstring.
#
# --cpus-per-task=8 (not 16, unlike the AdapTrans-only torch-order job):
# this worker DOES cache per_seq, but keyed by (cf_index, pref_dur_ms,
# sigma_dur_ms) -- up to 30 CF x 8 pref_dur x 8 k_sigma = 1,920 cache
# entries per worker within one alpha's pool, vs. the AdapTrans-only
# worker's ~30-entry cache (keyed by cf_index alone). Given the duration-
# only job's OOM-adjacent stall at 16 workers with NO caching at all (see
# batch_kim2024_style_duration_recovery_per_alpha.slurm's comment), and
# this worker's substantially larger per-worker cache footprint than that
# one, halving workers up front here is the conservative choice until
# proven otherwise at full scale.

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

# --no-capture-output matters for live progress monitoring -- see
# batch_kim2024_style_recovery_torch_order_per_alpha.slurm's comment.
#
# NOTE: if solver/ was added to this checkout after strf3.9's last editable
# install, rerun: conda run -n strf3.9 pip install -e .
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_recovery_torch_order_combined_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
