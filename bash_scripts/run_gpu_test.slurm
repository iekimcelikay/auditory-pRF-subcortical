#!/bin/bash
#SBATCH --job-name=prf_gpu_test
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=prf_gpu_test.%j.out
#SBATCH --error=prf_gpu_test.%j.err

module load CUDA/12.4.0

conda activate subcorticalSTRF3.9

# Install CUDA PyTorch if not already done (first time only)
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify GPU visible
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"

# Run the test suite (test 7 will now execute instead of skipping)
python tests/test_torch_pipeline.py
