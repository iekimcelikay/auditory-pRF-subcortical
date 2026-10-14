#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=kim2024_style_duration_recovery_torch_order_per_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_%j.out
#SBATCH --error=/scratch/ecelikay/workspace/auditory-pRF-subcortical/outputs_terminal/kim2024_style_duration_recovery_torch_order_per_alpha_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.celikay@bcbl.eu

# Duration-only recovery test using torch_pipeline.py's stage order + fixed
# per-CF reference calibration -- see
# kim2024_style_duration_recovery_torch_order.py's module docstring.
#
# --cpus-per-task=8 (not 16): this worker does no per-candidate caching
# (per_seq depends on cf_index+alpha+pref_dur+sigma_dur jointly here), and
# the non-torch-order duration job (job 7674151/7674463) silently stalled
# with zero surviving worker processes at 16 workers on this exact kind of
# uncached worker -- see batch_kim2024_style_duration_recovery_per_alpha.slurm's
# comment for the full incident. Halving workers up front here rather than
# rediscovering the same failure mode again.

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
conda run --no-capture-output -n strf3.9 python -m auditory_prf.prf_pipeline.solver.kim2024_style_duration_recovery_torch_order_parallel_per_alpha

echo "==================================="
echo "Job finished at $(date)"
echo "==================================="
