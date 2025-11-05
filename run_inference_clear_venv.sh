#!/bin/bash
#SBATCH --job-name=qoq_med_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules - PyTorch with CUDA 12.1 and GCC 11
echo "Loading Python 3.11 and PyTorch environment with CUDA 12.1 and GCC 11..."
module load python311
module load pytorch-py311-cuda12.1-gcc11/2.2.0

# Verify module setup
echo "Loaded modules:"
module list
echo "Python version:"
python3 --version
echo "gcc version:"
gcc --version
echo "nvcc version:"
nvcc --version 2>/dev/null || echo "nvcc not in PATH (this is OK if using prebuilt wheels)"

# Remove old virtual environment to ensure clean install
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (required by other packages)
echo "Installing PyTorch for CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install build dependencies for flash-attn
echo "Installing build dependencies..."
pip install ninja packaging wheel setuptools psutil

# Install flash-attn with no build isolation
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

# Install other dependencies from requirements.txt
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Display GPU information
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Run the inference script
echo "Running inference script..."
python simple_inference_example.py

# Print completion time
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="