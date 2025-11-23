# Image Denoising Project

A comprehensive toolkit for image denoising using traditional and deep learning methods.

## Project Structure

```
image_denoising/
│
├── data/
│   ├── noisy/          # Noisy input images
│   └── clean/          # Clean ground truth images
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

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
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

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, typical range 20-40 dB
- **SSIM** (Structural Similarity Index): Range [-1, 1], 1 means identical
- **MSE** (Mean Squared Error): Lower is better

## Examples

### Example 1: Quick comparison
```bash
python main.py -n ./gaussian_noise_15_sigma -c ./clean_images --compare --visualize --num-images 3
```

### Example 2: Denoise and save with Wiener filter
```bash
python main.py -n ./gaussian_noise_15_sigma -c ./clean_images -m wiener -o ./denoised_output
```

### Example 3: Batch evaluation
```bash
python evaluate.py -n ./gaussian_noise_15_sigma -c ./clean_images -o evaluation_results.csv
```

## Wiener Filter Visualization

The Wiener filter provides additional frequency domain visualization showing:
- Fourier Transform (log magnitude) of the noisy image
- Wiener filter response in frequency domain

This helps understand how the filter attenuates different frequency components.

## Future Work

- [ ] Implement deep learning methods (DnCNN, U-Net, etc.)
- [ ] Add BM3D implementation
- [ ] Add training pipeline for deep models
- [ ] Add more noise types (Poisson, speckle)
- [ ] Add real-world blind denoising

## References

- Traditional denoising: OpenCV documentation
- Wiener filter: scipy.signal.wiener
- Metrics: scikit-image

## License

MIT License
