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

All images should be placed in the `data/` directory, organized into two folders:

- **`data/clean/`** - Ground truth images without noise
- **`data/noisy/`** - Noisy versions of the clean images

### Types of Clean Images

We use three different types of clean images to test denoising performance:

1. **Lorem Ipsum Text Images**
   - Generated using `Make_Images.py`
   - Black/white text on colored backgrounds
   - Various fonts, sizes, and colors
   - Good for testing edge preservation and readability after denoising
   - Sharp features make it easy to see artifacts

2. **X-Ray Medical Images**
   - Real-world medical imaging data
   - Grayscale with varying contrast levels
   - Contains fine details and textures typical in diagnostic imaging
   - Tests practical applications in medical image processing

3. **Geometric Shapes**
   - Simple shapes (circles, rectangles, triangles, etc.)
   - Clean edges and uniform colors
   - Ideal for quantitative analysis of edge preservation
   - Provides clear baseline for measuring denoising accuracy

You can add noise to clean images using `Make_Noise.py` which supports both Gaussian and Salt & Pepper noise types.

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

### Basic Denoising

```bash
# Denoise with a specific method
python main.py -n data/noisy -c data/clean -m bilateral --visualize

# Denoise with all methods and save results
python main.py -n data/noisy -c data/clean -m all -o output/
```

### Method Comparison

```bash
# Compare all methods with metrics and visualization
python main.py -n data/noisy -c data/clean --compare --visualize

# Compare specific number of images
python main.py -n data/noisy -c data/clean --compare --num-images 10
```

### Batch Evaluation

```bash
# Evaluate all methods and save results to CSV
python evaluate.py -n data/noisy -c data/clean -o results.csv

# Evaluate specific methods
python evaluate.py -n data/noisy -c data/clean --methods gaussian bilateral wiener
```

### Noise Estimation

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

```
Main Script (main.py):
  -n, --noisy PATH       Path to noisy images folder (required)
  -c, --clean PATH       Path to clean images folder (required)
  -m, --method METHOD    Denoising method: gaussian, median, bilateral, nlm, wiener, all
  -o, --output PATH      Output folder for denoised images
  --num-images N         Number of images to process (default: 5)
  --compare              Compare all methods with metrics
  --visualize            Show visualization plots
  --estimate-noise       Estimate noise levels

Evaluation Script (evaluate.py):
  -n, --noisy PATH       Path to noisy images folder (required)
  -c, --clean PATH       Path to clean images folder (required)
  -o, --output PATH      Output CSV file for results
  --methods [METHODS]    Methods to evaluate
```

## How We Measure Quality

- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality. Higher values are better, usually between 20-40 dB for decent results.
- **SSIM** (Structural Similarity Index): Compares structural information. Ranges from -1 to 1, where 1 means the images are identical.
- **MSE** (Mean Squared Error): Simple pixel-wise difference. Lower values indicate better denoising.

## Examples

### Example 1: Generate clean text images
```bash
python Make_Images.py -n 50 --width 800 --height 600 --output data/clean
```

### Example 2: Add noise to clean images
```bash
python Make_Noise.py -i data/clean -o data/noisy -t gaussian -s 25
```

### Example 3: Quick comparison
```bash
python main.py -n data/noisy -c data/clean --compare --visualize --num-images 3
```

### Example 4: Denoise and save with Wiener filter
```bash
python main.py -n data/noisy -c data/clean -m wiener -o ./denoised_output
```

### Example 5: Batch evaluation
```bash
python evaluate.py -n data/noisy -c data/clean -o evaluation_results.csv
```

## Wiener Filter Visualization

The Wiener filter provides additional frequency domain visualization showing:
- Fourier Transform (log magnitude) of the noisy image
- Wiener response in frequency domain

This helps understand how the filter attenuates different frequency components.

## Future Work

- [ ] Implement deep learning methods (DnCNN, U-Net, etc.)
- [ ] Add BM3D implementation
- [ ] Add training pipeline for deep models
- [ ] Add more noise types (Salt n Pepper/Motion Blur)

## References

- Traditional denoising: OpenCV documentation
- Wiener filter: scipy.signal.wiener
- Metrics: scikit-image

