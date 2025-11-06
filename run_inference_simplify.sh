#!/bin/bash
#SBATCH --job-name=qoq_med_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=interactive
#SBATCH --nodelist=gpu317
#SBATCH --gres=gpu:1
#SBATCH --mail-user=xhan56@jh.edu
#SBATCH --mail-type=end

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

module load python311
module load pytorch-py311-cuda12.1-gcc11/2.2.0

echo "Loaded modules:"
module list
echo "Python version:"
python3 --version
echo "gcc version:"
gcc --version

# Set up Python environment
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install flash-attn from prebuilt wheel to avoid cross-device link error
echo "Installing flash-attn from prebuilt wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl || {
    echo "WARNING: Failed to install flash-attn from prebuilt wheel"
    echo "Trying alternative installation method..."
    pip install flash-attn --no-cache-dir --no-build-isolation || {
        echo "WARNING: flash-attn installation failed. Will use eager attention instead."
    }
}

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check GPU availability
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Run the inference script
echo "Running simple_inference_example.py..."
python simple_inference_example.py

# Print completion time
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=========================================="