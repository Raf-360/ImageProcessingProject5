# Image Denoising: Traditional vs Deep Learning Methods

A comprehensive comparison study of traditional image denoising techniques versus modern deep learning approaches (DnCNN) for removing multiple types of noise from synthetic checkered triangle images.

## Project Overview

This project evaluates and compares the performance of classical filtering methods with state-of-the-art deep learning models for image denoising tasks. The study focuses on:

- **Traditional Methods**: Median filtering, Gaussian filtering, bilateral filtering, non-local means, etc.
- **Deep Learning**: DnCNN (Denoising Convolutional Neural Network)
- **Noise Types**: 
  - Salt-and-pepper impulse noise at various densities (2%, 10%)
  - Gaussian additive noise at various standard deviations (σ = 10, 15, 25)
- **Evaluation Metrics**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)
- **Visualization Tools**: Noise distribution analysis, statistical validation (Q-Q plots)

## Dataset

### Synthetic Image Generation
- **Base Images**: Checkered triangle patterns with varying colors and rotation angles
- **Image Size**: 300×300 pixels
- **Rotations**: 0° to 120° in 5° increments
- **Color Variations**: Multiple foreground/background color combinations

### Noise Addition
- **Noise Types**: 
  1. **Salt-and-pepper impulse noise**
     - Randomly replaces pixels with pure white (salt) or black (pepper)
     - Densities: 2% (sparse), 10% (moderate)
     - Ground truth CSV files contain exact (x, y) coordinates of each impulse
  2. **Gaussian additive noise**
     - Adds random values drawn from normal distribution N(0, σ²)
     - Standard deviations: σ = 10, 15, 25
     - Zero-mean noise with controlled variance
     - Affects all pixels uniformly across color channels
- **Total Images**: Several thousand noisy images with corresponding noise maps/parameters

## Project Structure

```
Project5/
├── README.md                                    # This file
├── Make_Images.py                               # Generates clean checkered triangle images
├── Make_Noise.py                                # Adds salt-and-pepper noise and logs coordinates
├── visualization.py                             # Gaussian noise analysis and visualization CLI tool
├── traditional_filtering.py                     # Traditional denoising implementations
├── requirements.txt                             # Python dependencies
├── output_images/                               # Clean original images
├── gaussian_noise_10_sigma/                     # Gaussian noise (σ=10) dataset
│   └── noisy_*.png                             # Noisy images
├── gaussian_noise_15_sigma/                     # Gaussian noise (σ=15) dataset
│   └── noisy_*.png                             # Noisy images
├── Noise_Images_Salt_And_Pepper_2_percent/     # 2% noise density dataset
│   ├── noisy_*.png                             # Noisy images
│   └── noisy_*.csv                             # Noise coordinate maps
├── Noise_Images_Salt_And_Pepper_10_Percent/    # 10% noise density dataset
│   ├── noisy_*.png                             # Noisy images
│   └── noisy_*.csv                             # Noise coordinate maps
└── Proj5_2025.pdf                              # Project specifications
```

## Methodology

### 1. Data Preparation
- Generate clean synthetic images with controlled patterns
- Add multiple noise types at specified parameters:
  - Salt-and-pepper: Record ground truth locations in CSV format
  - Gaussian: Apply zero-mean additive noise with specified σ values
- Validate noise characteristics (distribution, mean, variance)

### 2. Traditional Filtering Methods
Implementation of classical denoising techniques:

**For Salt-and-Pepper Noise:**
- **Median Filter**: Non-linear filter effective for impulse noise
- **Adaptive Median Filter**: Variable window size based on local statistics
- **Morphological Filters**: Opening/closing operations

**For Gaussian Noise:**
- **Gaussian Filter**: Linear smoothing filter
- **Bilateral Filter**: Edge-preserving smoothing
- **Non-Local Means**: Patch-based denoising
- **Wiener Filter**: Optimal linear filter for AWGN

**Universal Techniques:**
- Can be applied to both noise types with varying effectiveness

### 3. Deep Learning Method (DnCNN)
- Pre-trained or custom-trained DnCNN model
- Convolutional neural network architecture
- Residual learning approach
- Training on large datasets

### 4. Performance Evaluation
- **PSNR**: Measures reconstruction quality (higher is better)
- **SSIM**: Measures structural similarity (0-1, higher is better)
- **Visual Quality**: Subjective assessment of denoised images
- **Execution Time**: Processing speed comparison
- **Resource Requirements**: Memory, CPU/GPU usage

## Expected Comparisons

| Aspect | Traditional Methods | Deep Learning (DnCNN) |
|--------|--------------------|-----------------------|
| **Development** | Simple implementation | Complex architecture, training pipeline |
| **Dataset Requirements** | None (model-free) | Large training dataset required |
| **Training Time** | N/A | Hours to days (GPU-dependent) |
| **Inference Speed** | Fast (CPU) | Moderate (GPU) to slow (CPU) |
| **Hardware Demands** | Low (CPU sufficient) | High (GPU recommended) |
| **Adaptability** | Fixed algorithm | Learns from data |
| **Performance** | Good for specific noise | Excellent with proper training |

## Usage

### Generate Clean Images
```bash
python Make_Images.py
```

### Add Noise to Images
```bash
# For salt-and-pepper noise
python Make_Noise.py

# For Gaussian noise (modify parameters in script)
# Adjust sigma values: 10, 15, 25
```

### Analyze Gaussian Noise Distribution
```bash
# Analyze first 3 image pairs
python visualization.py -o ./output_images -n ./gaussian_noise_15_sigma -s 3

# Analyze all image pairs
python visualization.py -o ./output_images -n ./gaussian_noise_15_sigma

# Non-interactive mode (auto-advance through all images)
python visualization.py -o ./output_images -n ./gaussian_noise_10_sigma --no-interactive

# Get help
python visualization.py --help
```

**Visualization Features:**
- Side-by-side comparison of original, noisy, and noise-only images
- Per-channel noise histograms with Gaussian curve fitting
- Combined noise distribution analysis
- Q-Q plots for normality testing
- Comprehensive noise statistics (mean, std dev, min, max per channel)
- Statistical validation of Gaussian properties (μ ≈ 0)

### Run Traditional Filters
```bash
python traditional_filtering.py
```

### Load Noise Coordinates
```python
import csv

def load_noise_coordinates(csv_path):
    """Load salt and pepper noise coordinates from CSV file."""
    salt_coords = []
    pepper_coords = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = int(row['X']), int(row['Y'])
            if row['Type'] == 'Salt':
                salt_coords.append((x, y))
            else:
                pepper_coords.append((x, y))
    
    return salt_coords, pepper_coords
```

## Requirements

```
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
matplotlib>=3.4.0
scipy>=1.7.0
```

For deep learning methods:
```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0 (alternative)
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## Results

Results will include:

### Noise Analysis
- Validation of Gaussian noise properties (zero mean, specified variance)
- Visual confirmation of noise distribution through histograms
- Q-Q plots demonstrating normality of Gaussian noise
- Per-channel and combined noise statistics

### Denoising Performance
- Side-by-side visual comparisons (original, noisy, denoised)
- Quantitative metrics table (PSNR/SSIM per method and noise type)
- Performance comparison: salt-and-pepper vs Gaussian noise
- Method effectiveness analysis (which filters work best for which noise)
- Execution time benchmarks
- Discussion of trade-offs

### Key Findings
- Median filters excel at salt-and-pepper but fail with Gaussian noise
- Gaussian/bilateral filters effective for AWGN but poor for impulse noise
- DnCNN performance depends on training data diversity
- Computational cost vs quality trade-offs

## Future Work

- Extend to other noise types (Poisson, speckle, mixed noise)
- Test on real-world images (medical, satellite, photography)
- Explore other deep learning architectures (U-Net, RED-Net, Restormer)
- Implement hybrid approaches (traditional + deep learning)
- Real-time denoising applications
- Blind denoising (unknown noise type/parameters)
- Multi-scale and multi-resolution techniques
- Noise estimation algorithms

## References

- Zhang, K., et al. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." IEEE TIP
- Buades, A., Coll, B., & Morel, J. M. (2005). "A non-local algorithm for image denoising." CVPR
- Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for gray and color images." ICCV
- Dabov, K., et al. (2007). "Image denoising by sparse 3-D transform-domain collaborative filtering." IEEE TIP
- Foi, A., et al. (2007). "Pointwise Shape-Adaptive DCT for High-Quality Denoising." IEEE TIP


## Authors

Rafael Moreno
Alejandro Rubio
Carlos Lopez
Fall 2025 - TTU ECE 4367 Image Processing
