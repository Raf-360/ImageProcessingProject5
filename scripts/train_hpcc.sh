#!/bin/bash
#SBATCH --job-name=dncnn_train       # Job name
#SBATCH --output=logs/train_%j.out   # Standard output (%j = job ID)
#SBATCH --error=logs/train_%j.err    # Standard error
#SBATCH --partition=gpu              # GPU partition
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # CPUs for data loading
#SBATCH --mem=32G                    # Memory
#SBATCH --time=24:00:00              # Max runtime (24 hours)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=alejrubi@ttu.edu 

# Exit on any error
set -e

# Print job information
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================================"

# Navigate to project directory
PROJECT_DIR="/lustre/work/$USER/Project5"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# Load modules
module purge
module load gcc/11.2.0
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.10

echo "Loaded modules:"
module list

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip first
pip install --upgrade pip

# Install dependencies if needed
if [ ! -f "$VENV_DIR/.installed" ]; then
    echo "Installing dependencies..."
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    pip install pyyaml pillow numpy scikit-image matplotlib tensorboard tqdm
    
    # Mark as installed
    touch "$VENV_DIR/.installed"
else
    echo "Dependencies already installed."
fi

# Verify GPU availability
echo "================================================================"
echo "GPU Information:"
nvidia-smi
echo "================================================================"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "================================================================"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p tensorboard_logs

# Set CUDA environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Print training configuration
echo "Training configuration:"
cat configs/dncnn_train.yaml
echo "================================================================"

# Run training with error handling
echo "Starting training..."
if python train.py --config configs/dncnn_train.yaml; then
    echo "Training completed successfully!"
    EXIT_CODE=0
else
    echo "Training failed with error code $?"
    EXIT_CODE=1
fi

# Print completion info
echo "================================================================"
echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

# Save GPU usage stats
nvidia-smi > logs/gpu_stats_$SLURM_JOB_ID.txt

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
