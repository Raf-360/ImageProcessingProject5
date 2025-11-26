# Image Denoising Project

This project implements various image denoising techniques, ranging from traditional filters to modern deep learning approaches. Built as part of my computer vision coursework.

## Project Structure

```
image_denoising/
│
├── data/
│   ├── noisy/          # Noisy input images (all noise types)
│   └── clean/          # Clean ground truth images (all types)
│
├── traditional/        # Traditional denoising methods
│   ├── gaussian.py     # Gaussian blur filter
│   ├── median.py       # Median filter
│   ├── bilateral.py    # Bilateral filter
│   ├── nlm.py          # Non-Local Means
│   ├── wiener.py       # Wiener filter
│   └── bm3d.py         # BM3D (placeholder)
│
├── deep/               # Deep learning methods (future)
│   ├── dncnn.py        # DnCNN
│   ├── unet_denoiser.py
│   ├── autoencoder.py
│   ├── diffusion.py
│   └── utils.py
│
├── utils/              # Utility functions
│   ├── metrics.py      # PSNR, SSIM, MSE
│   ├── visualization.py # Plotting functions
│   ├── image_io.py     # Image loading/saving
│   └── noise_estimation.py # Noise level estimation
│
├── configs/            # Configuration files
│
├── main.py             # Main CLI entry point
├── evaluate.py         # Batch evaluation script
└── README.md
```

## Dataset Organization

All images are organized in the `data/` directory with proper train/test/validation splits for deep learning:

```
data/
├── train/ (80%)          # Training data
│   ├── xray/            # Medical X-ray images (primary focus)
│   ├── synthetic/       # Text and geometric shapes
│   └── natural/         # Natural scene images
├── test/ (10%)          # Test data (same structure)
└── validation/ (10%)    # Validation data (same structure)
```

Each category contains:
- `clean/` or `*_images/` - Ground truth clean images
- `gaussian_noise_*_sigma/` - Noisy versions at different sigma levels

### Types of Images

1. **X-Ray Medical Images** (Primary Focus)
   - 5000 medical X-ray images
   - Grayscale with varying contrast levels
   - Fine details and textures typical in diagnostic imaging
   - Noise levels: σ=15, σ=55
   - Critical for medical image processing applications

2. **Synthetic Images** (Text & Shapes)
   - **Lorem Ipsum Text Images**
     - Black/white text on colored backgrounds
     - Various fonts, sizes, and colors
     - Good for testing edge preservation and readability
   - **Geometric Shapes**
     - Simple shapes (circles, rectangles, triangles)
     - Clean edges and uniform colors
     - Ideal for quantitative edge preservation analysis
   - Noise levels: σ=15, σ=25, σ=50

3. **Natural Images** (Ready for Expansion)
   - Currently empty, ready for natural scene images
   - Will provide texture diversity for training
   - Complements medical and synthetic data

### Data Split Strategy

- **80% Training:** Used to train DnCNN model
- **10% Test:** Final evaluation, never seen during training
- **10% Validation:** Hyperparameter tuning and early stopping

Images are split consistently across all noise levels to maintain correspondence between clean/noisy pairs.

## Getting Started

Set up your environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows users: .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

## Usage

> **Note:** This project has three main scripts with different purposes:
> - **`main.py`** - For denoising and quick visualization  
> - **`generate_report.py`** - For comprehensive reports, error maps, PDFs, HTML
> - **`evaluate.py`** - For batch evaluation to CSV

### Basic Denoising (main.py)

```bash
# Denoise with a specific method
python main.py -n data/noisy -c data/clean -m bilateral --visualize

# Denoise with all methods and save results
python main.py -n data/noisy -c data/clean -m all -o output/

# Apply filter multiple times iteratively
python main.py -n data/noisy -c data/clean -m median --iterations 3 --visualize
```

### Auto-Tuning Parameters (main.py)

```bash
# Auto-tune parameters to find best PSNR
python main.py -n data/noisy -c data/clean -m bilateral --auto-tune

# Auto-tune for best SSIM with 20 iterations
python main.py -n data/noisy -c data/clean -m nlm --auto-tune --tune-metric ssim --tune-iterations 20

# Auto-tune with filter iterations optimization
python main.py -n data/noisy -c data/clean -m median --auto-tune --tune-iterations 30
```

### Method Comparison (main.py)

```bash
# Compare all methods with metrics and visualization
python main.py -n data/noisy -c data/clean --compare --visualize

# Compare specific number of images
python main.py -n data/noisy -c data/clean --compare --num-images 10

# Visualize multiple images with navigation (press q to continue)
python main.py -n data/noisy -c data/clean -m bilateral --visualize --num-images 5
```

### Comprehensive Reports (generate_report.py)

```bash
# Generate complete analysis report with all visualizations
python generate_report.py -n data/noisy -c data/clean --all

# Generate only error maps
python generate_report.py -n data/noisy -c data/clean --error-maps --num-images 5

# Generate only dataset-wide analysis plots
python generate_report.py -n data/noisy -c data/clean --dataset-plots

# Generate HTML report
python generate_report.py -n data/noisy -c data/clean --html-report

# Generate PDF report
python generate_report.py -n data/noisy -c data/clean --pdf-report
```

### Batch Evaluation (evaluate.py)

```bash
# Evaluate all methods and save results to CSV
python evaluate.py -n data/noisy -c data/clean -o results.csv

# Evaluate specific methods
python evaluate.py -n data/noisy -c data/clean --methods gaussian bilateral wiener
```

### Noise Estimation (main.py)

```bash
# Estimate noise levels in images
python main.py -n data/noisy -c data/clean --estimate-noise
```

## Available Methods

### Traditional Methods

1. **Gaussian Blur** (`gaussian`)
   - Fast, simple baseline
   - Blurs edges
   - Parameters: kernel_size, sigma

2. **Median Filter** (`median`)
   - Best for salt-and-pepper noise
   - Preserves edges
   - Parameters: kernel_size

3. **Bilateral Filter** (`bilateral`)
   - Edge-preserving smoothing
   - Good for Gaussian noise
   - Parameters: d, sigma_color, sigma_space

4. **Non-Local Means** (`nlm`)
   - State-of-the-art traditional method
   - Excellent for Gaussian noise
   - Parameters: h, template_window_size, search_window_size

5. **Wiener Filter** (`wiener`)
   - Optimal linear filter
   - Frequency domain filtering
   - Parameters: mysize, noise_variance

## Command Line Options

### Main Script (main.py)
```
  -n, --noisy PATH          Path to noisy images folder (required)
  -c, --clean PATH          Path to clean images folder (required)
  -m, --method METHOD       Denoising method: gaussian, median, bilateral, nlm, wiener, all
  -o, --output PATH         Output folder for denoised images
  --num-images N            Number of images to process (default: 5)
  --iterations N            Apply filter N times iteratively (default: 1)
  --compare                 Compare all methods with metrics
  --visualize               Show visualization plots
  --estimate-noise          Estimate noise levels
  --auto-tune               Use Bayesian optimization to find best parameters
  --tune-metric METRIC      Metric to optimize: psnr or ssim (default: psnr)
  --tune-iterations N       Number of optimization iterations (default: 15)
```

### Report Generation Script (generate_report.py)
```
  -n, --noisy PATH          Path to noisy images folder (required)
  -c, --clean PATH          Path to clean images folder (required)
  -o, --output PATH         Output folder for reports (default: ./reports)
  --num-images N            Number of images to process (default: 5)
  --error-maps              Generate error maps for each method
  --dataset-plots           Generate dataset-wide analysis plots
  --html-report             Generate HTML report with embedded visualizations
  --pdf-report              Generate PDF report with all plots
  --all                     Generate all reports (error maps, plots, HTML, PDF)
```

### Evaluation Script (evaluate.py)
```
  -n, --noisy PATH          Path to noisy images folder (required)
  -c, --clean PATH          Path to clean images folder (required)
  -o, --output PATH         Output CSV file for results
  --methods [METHODS]       Methods to evaluate
```

## How We Measure Quality

- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality. Higher values are better, usually between 20-40 dB for decent results.
- **SSIM** (Structural Similarity Index): Compares structural information. Ranges from -1 to 1, where 1 means the images are identical.

## Examples

### Example 1: Generate clean text images
```bash
python Make_Images.py -n 50 --width 800 --height 600 --output data/clean
```

### Example 2: Add noise to clean images
```bash
python Make_Noise.py -i data/clean -o data/noisy -t gaussian -s 25
```

### Example 3: Quick comparison with visualization
```bash
python main.py -n data/noisy -c data/clean --compare --visualize --num-images 3
```

### Example 4: Auto-tune bilateral filter for best SSIM
```bash
python main.py -n data/noisy -c data/clean -m bilateral --auto-tune --tune-metric ssim
```

### Example 5: Apply median filter 3 times iteratively
```bash
python main.py -n data/noisy -c data/clean -m median --iterations 3 --visualize
```

### Example 6: Denoise and save with Wiener filter
```bash
python main.py -n data/noisy -c data/clean -m wiener -o ./denoised_output
```

### Example 7: Batch evaluation to CSV
```bash
python evaluate.py -n data/noisy -c data/clean -o evaluation_results.csv
```

### Example 8: Generate comprehensive analysis report
```bash
python generate_report.py -n data/noisy/gaussian_noise_25_sigma -c data/clean/clean_images --all
```

### Example 9: Generate only error maps with noise distribution
```bash
python generate_report.py -n data/noisy -c data/clean --error-maps --num-images 5
```

### Example 10: Multi-image visualization (press q to navigate)
```bash
python main.py -n data/noisy -c data/clean -m nlm --visualize --num-images 10
```

## Wiener Filter Visualization

The Wiener filter provides additional frequency domain visualization showing:
- Fourier Transform (log magnitude) of the noisy image
- Wiener response in frequency domain

This helps understand how the filter attenuates different frequency components.

## Project Goals & Progress

### ✅ Completed

**1. Complete Image Denoising Benchmarking Framework**
- ✅ Unified loader for noisy and clean image pairs (`utils/image_io.py`)
- ✅ Unified interface for denoising methods
- ✅ Evaluation pipeline with PSNR, SSIM, MSE (`utils/metrics.py`)
- ✅ Visualization functions (`utils/visualization.py`)
- ✅ CLI to run experiments (`main.py`, `evaluate.py`)

**2. Traditional Image Denoising Methods**
- ✅ Gaussian blur with iterative support
- ✅ Median filtering with iterative support
- ✅ Bilateral filtering with iterative support
- ✅ Non-local means (NLM)
- ✅ Wiener filtering with FFT visualization
- ✅ Bayesian parameter auto-tuning (`--auto-tune`)
- ✅ Noise estimation (MAD method)
- ✅ Batch processing support

**3. Evaluation Metrics**
- ✅ PSNR, SSIM, MSE
- ✅ Runtime measurements
- ✅ Robustness testing across noise levels

**4. Visualization & Reporting**
- ✅ Side-by-side comparisons
- ✅ FFT visualizations (Wiener filter)
- ✅ Multi-image visualization with navigation
- ✅ Error maps (`utils/error_maps.py`)
- ✅ Dataset-wide plots (`utils/dataset_plots.py`)
- ✅ PDF/HTML report generation (`utils/report_generation.py`)

**5. CLI Features**
- ✅ Run specific methods (`-m/--method`)
- ✅ Benchmark all methods (`--compare`)
- ✅ Auto-tune parameters (`--auto-tune`)
- ✅ Iterative filtering (`--iterations`)
- ✅ Save outputs (`-o/--output`)
- ✅ Noise estimation (`--estimate-noise`)
- ✅ Visualization (`--visualize`)

**6. Modular Architecture**
- ✅ Separate directories for traditional, utils, deep, configs
- ✅ Clean separation of concerns
- ✅ Plug-in structure for adding new methods

### ⏳ In Progress

**7. Deep Learning Denoising Models - DnCNN Training**

**Current Status:** Preparing dataset and infrastructure for DnCNN training

**What We're Doing Now:**
- ✅ Dataset preparation with proper splits (80% train / 10% test / 10% validation)
- ✅ Adding Gaussian noise at multiple sigma levels (σ=15, 25, 55) to XRAY images
- ✅ Organizing data into structured folders: `data/train/`, `data/test/`, `data/validation/`
- ✅ Maintaining XRAY-focused distribution (medical imaging priority)
- ✅ Including synthetic images (shapes, text) and natural images for diversity
- 🔄 Creating data reorganization utilities (`deep/utils/move_data.py`)

**Dataset Composition:**
- **XRAY Images** (Largest category, primary focus)
  - 5000 medical X-ray images
  - Noise levels: σ=15, σ=55
  - Location: `data/train/xray/`
  
- **Synthetic Images** (Text & shapes)
  - Lorem ipsum text images with varying fonts/colors
  - Geometric shapes with clean edges
  - Noise levels: σ=15, σ=25, σ=50
  - Location: `data/train/synthetic/`
  
- **Natural Images** (Empty for now, ready for expansion)
  - Location: `data/train/natural/`

**Next Steps (In Order):**
1. 🔄 Finalize data reorganization script and execute split
2. 📋 Install PyTorch and dependencies (`torch`, `torchvision`)
3. 📋 Implement DnCNN architecture (17-layer CNN with residual learning)
   - Create `deep/dncnn.py` with model definition
   - Implement residual noise learning (predict noise, not clean image)
4. 📋 Create PyTorch Dataset class (`deep/dataset.py`)
   - Wrap existing image loaders from `utils/image_io.py`
   - Extract 40×40 patches for training
   - Implement data augmentation (flips, rotations)
5. 📋 Build training infrastructure (`train.py` or `deep/train_dncnn.py`)
   - Training loop with GPU support
   - Validation loop using existing metrics (PSNR/SSIM)
   - Checkpoint management and model saving
   - TensorBoard logging for loss curves
   - Learning rate scheduling
6. 📋 Create configuration system (`configs/dncnn_train.yaml`)
   - Model hyperparameters (depth, channels, batch size)
   - Training parameters (learning rate, epochs, optimizer)
   - Data paths and augmentation settings
7. 📋 Implement inference wrapper
   - Create `dncnn_denoise()` function matching traditional method signatures
   - Integrate with existing `main.py` CLI (add `-m dncnn` option)
8. 📋 Training and evaluation
   - Train on σ=25 Gaussian noise (primary noise level)
   - Validate generalization on σ=15 and σ=55
   - Compare DnCNN vs traditional methods (bilateral, NLM, BM3D)
   - Generate comprehensive reports with `generate_report.py`

**Technical Details:**
- **Model:** DnCNN-S (17 convolutional layers)
  - Architecture: Conv + ReLU → (Conv + BatchNorm + ReLU) × 15 → Conv
  - Residual learning: Network predicts noise, not clean image
  - Formula: `Clean = Noisy - Predicted_Noise`
- **Training:** 
  - Loss: MSE between predicted and actual noise
  - Optimizer: Adam with learning rate decay
  - Batch size: 128 patches (40×40)
  - Data augmentation: Random flips, rotations
- **Evaluation:** PSNR, SSIM on held-out test set

**8. Configurable Experiment System**
- 📋 YAML/JSON configs for reproducibility (directory created, implementing next)

### 📋 Planned

**9. Advanced DnCNN Features**
- 📋 Multi-noise-level training (blind denoising model)
- 📋 Color image support (currently grayscale-focused)
- 📋 Real-time inference optimization
- 📋 Model ensemble for improved performance
- 📋 Transfer learning from pre-trained models

**10. Additional Deep Learning Models**
- 📋 UNet-based denoiser (encoder-decoder architecture)
- 📋 Denoising autoencoder (latent space learning)
- 📋 Diffusion models (state-of-the-art generative approach)
- 📋 Self-supervised denoisers (Noise2Noise, Noise2Void)

**11. Advanced Features & Datasets**
- 📋 Real-noise datasets (SIDD, DND) for practical evaluation
- 📋 Additional noise types (Poisson, speckle, motion blur)
- 📋 LPIPS perceptual metric for quality assessment
- 📋 Mixed precision training (faster on modern GPUs)

**12. Reporting & Comparison**
- 📋 Traditional vs deep learning performance study
- 📋 Computational efficiency analysis (speed vs quality)
- 📋 Strengths and weaknesses analysis per method
- 📋 Domain-specific performance (medical vs natural vs synthetic)

**13. Stretch Goals**
- 📋 Optional GUI interface for interactive denoising
- 📋 Web app demo with model deployment
- 📋 Publish as open-source package
- 📋 Real-time video denoising demo
- 📋 Integration with medical imaging pipelines

## References

- Traditional denoising: OpenCV documentation
- Wiener filter: scipy.signal.wiener
- Metrics: scikit-image
- Bayesian optimization: scikit-optimize

