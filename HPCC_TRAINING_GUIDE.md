# DnCNN Training on TTU HPCC

Complete guide for training DnCNN image denoising model on TTU's High Performance Computing Center.

## 📋 Prerequisites

- TTU HPCC account 
- SSH access to HPCC
- VPN Access to HPCC (For off campus)
- Data already organized in `/lustre/work/username/Project5/data/`

## 🚀 Quick Start

### 1. Connect to HPCC

```bash
ssh username@login.hpcc.ttu.edu
```

### 2. Navigate to Project Directory

```bash
cd /lustre/work/username/Project5
```

### 3. Prepare Data (if not already done)

If data is not organized into train/test/validation splits:

```bash
# From your local machine, copy data to HPCC
scp -r data/ username@login.hpcc.ttu.edu:/lustre/work/username/Project5/

```

Expected structure after organization:
```
/scratch/username/Project5/data/
├── train/
│   ├── xray/
│   │   ├── clean/
│   │   ├── gaussian_noise_15_sigma/
│   │   ├── gaussian_noise_25_sigma/
│   │   └── gaussian_noise_55_sigma/
│   ├── synthetic/
│   └── jellyfish/
├── test/
└── validation/
```

### 4. Submit Training Job

```bash
# Make script executable
chmod +x scripts/train_hpcc.sh

# Submit job
sbatch scripts/train_hpcc.sh
```

### 5. Monitor Training

```bash
# Check job status
squeue -u username

# View live output (replace <job_id> with actual job ID)
tail -f logs/train_<job_id>.out

# View error log
tail -f logs/train_<job_id>.err
```

### 6. Monitor with TensorBoard (Optional)

On HPCC:
```bash
# Start TensorBoard on compute node
tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006
```

From your local machine:
```bash
# Create SSH tunnel (replace <node_name> with actual compute node)
ssh -L 6006:<node_name>:6006 username@login.hpcc.ttu.edu

# Open in browser: http://localhost:6006
```

## 📊 Training Configuration

Configuration file: `configs/dncnn_train.yaml`

**Key settings:**
- **Batch size:** 128
- **Epochs:** 50
- **Learning rate:** 0.001 (with ReduceLROnPlateau scheduler)
- **Early stopping:** Patience of 10 epochs
- **GPU:** 1 x NVIDIA GPU (auto-detected)
- **Data:** XRAY (primary), synthetic, jellyfish at σ=15, 25, 55

## 📁 Output Files

After training, you'll find:

```
/scratch/username/Project5/
├── checkpoints/
│   ├── checkpoint_best.pth          # Best model (highest validation PSNR)
│   ├── checkpoint_latest.pth        # Most recent checkpoint
│   └── checkpoint_epoch_*.pth       # Periodic checkpoints
├── tensorboard_logs/                # TensorBoard logs
└── logs/
    ├── train_<job_id>.out          # Training output
    └── train_<job_id>.err          # Error messages
```

## 🔍 Check Training Results

```bash
# View training summary
grep "Best PSNR" logs/train_*.out

# View final metrics
tail -n 50 logs/train_*.out

# List saved checkpoints
ls -lh checkpoints/
```

## 📥 Download Trained Model

From your local machine:

```bash
# Download best checkpoint
scp username@login.hpcc.ttu.edu:/lustre/work/username/Project5/checkpoints/checkpoint_best.pth ./

# Or download all checkpoints
scp -r username@login.hpcc.ttu.edu:/lustre/work/username/Project5/checkpoints ./
```

## 🧪 Local Testing (Before HPCC)

Test training pipeline locally with smaller dataset:

```bash
# Modify config for local testing
# In configs/dncnn_train.yaml, set:
#   - max_images: 100
#   - epochs: 5
#   - batch_size: 16

# Run training
python train.py --config configs/dncnn_train.yaml
```

## 🎯 Using Trained Model for Inference

Once you have a trained model:

```python
from deep.inference import dncnn_denoise
import cv2

# Load noisy image
noisy_img = cv2.imread('noisy_image.png', cv2.IMREAD_GRAYSCALE)

# Denoise
denoised_img = dncnn_denoise(noisy_img, checkpoint_path='checkpoints/checkpoint_best.pth')

# Save result
cv2.imwrite('denoised_image.png', denoised_img)
```

Or use with existing CLI:

```bash
python main.py -i data/noisy/test_image.png -o results/ -m dncnn
```

## ⚙️ SLURM Job Configuration

The `scripts/train_hpcc.sh` submission script uses:

- **Partition:** gpu
- **GPU:** 1 GPU
- **CPUs:** 8 (for data loading)
- **Memory:** 32GB
- **Time limit:** 24 hours
- **Email:** username@ttu.edu (notifications on completion/failure)

To modify resources, edit the `#SBATCH` directives in `scripts/train_hpcc.sh`.

## 🐛 Troubleshooting

### Job Pending

```bash
# Check queue
squeue -u username

# Check reasons
squeue -u username --start
```

### Out of Memory

If training crashes with OOM errors:
- Reduce batch size in `configs/dncnn_train.yaml`
- Request more memory in `scripts/train_hpcc.sh` (increase `--mem`)

### CUDA Not Available

Check error log for CUDA issues:
```bash
tail -100 logs/train_<job_id>.err
```

Verify modules in job script match HPCC's available modules:
```bash
module avail cuda
module avail cudnn
```

### Slow Training

- Check if GPU is being used: `nvidia-smi` in compute node
- Increase `num_workers` in config (currently 4)
- Verify data is in `/scratch` not `/home` (faster I/O)

## 📚 Model Architecture

**DnCNN Specifications:**
- **Layers:** 17 convolutional layers
- **Filters:** 64 filters per layer
- **Kernel size:** 5×5
- **Learning:** Residual learning (predicts noise, not clean image)
- **Parameters:** ~1.5M parameters (~6MB model size)

## 📞 Support

- **HPCC Documentation:** https://www.depts.ttu.edu/hpcc/
- **HPCC Support:** hpcc@ttu.edu
- **Training Issues:** Check logs in `logs/` directory

## ✅ Expected Training Time

With 1 GPU and ~5000 XRAY images:
- **Per epoch:** ~15-20 minutes
- **Total (50 epochs):** ~12-16 hours
- **Early stopping may finish earlier**

Monitor progress with `tail -f logs/train_<job_id>.out`
