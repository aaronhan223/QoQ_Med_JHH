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

# Load required modules (adjust based on your cluster's module system)
# Uncomment and modify as needed for your cluster
# module load python/3.10
# module load cuda/12.1

# Set up Python environment
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
# echo "Upgrading pip..."
# pip install --upgrade pip

# Install dependencies from requirements.txt
# echo "Installing dependencies from requirements.txt..."
# pip install -r requirements.txt

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
