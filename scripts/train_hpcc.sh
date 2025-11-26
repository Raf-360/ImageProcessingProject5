#!/bin/bash
#SBATCH --job-name=dncnn_train       # Job name
#SBATCH --output=logs/train_%j.out   # Standard output (%j = job ID)
#SBATCH --error=logs/train_%j.err    # Standard error
#SBATCH --partition=gpu              # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # CPUs for data loading
#SBATCH --mem=32G                    # Memory
#SBATCH --time=24:00:00              # Max runtime (24 hours)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=alejrubi@ttu.edu # Your email

# Print job information
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================"

# Navigate to project directory
cd /scratch/alejrubi/Project5 || exit 1

# Load modules
module purge
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.10

echo "Loaded modules:"
module list

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    touch venv/.installed
fi

# Verify GPU availability
echo "================================================================"
echo "GPU Information:"
nvidia-smi
echo "================================================================"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo "================================================================"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p tensorboard_logs

# Run training
echo "Starting training..."
python train.py --config configs/dncnn_train.yaml

# Print completion info
echo "================================================================"
echo "Training completed at: $(date)"
echo "================================================================"

# Deactivate virtual environment
deactivate
