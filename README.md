# Image Denoising: Traditional vs Deep Learning Methods

A comprehensive comparison study of traditional image denoising techniques versus modern deep learning approaches (DnCNN) for removing salt-and-pepper noise from synthetic checkered triangle images.

## Project Overview

This project evaluates and compares the performance of classical filtering methods with state-of-the-art deep learning models for image denoising tasks. The study focuses on:

- **Traditional Methods**: Median filtering, Gaussian filtering, bilateral filtering, non-local means, etc.
- **Deep Learning**: DnCNN (Denoising Convolutional Neural Network)
- **Noise Types**: Salt-and-pepper impulse noise at various densities (2%, 10%, etc.)
- **Evaluation Metrics**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)

## Dataset

### Synthetic Image Generation
- **Base Images**: Checkered triangle patterns with varying colors and rotation angles
- **Image Size**: 300×300 pixels
- **Rotations**: 0° to 120° in 5° increments
- **Color Variations**: Multiple foreground/background color combinations

### Noise Addition
- **Noise Type**: Salt-and-pepper impulse noise
- **Noise Densities**: 
  - 2% (sparse noise)
  - 10% (moderate noise)
- **Ground Truth**: CSV files containing exact (x, y) coordinates of each salt (white) and pepper (black) impulse
- **Total Images**: Several thousand noisy images with corresponding noise maps

## Project Structure

```
Project5/
├── README.md                                    # This file
├── Make_Images.py                               # Generates clean checkered triangle images
├── Make_Noise.py                                # Adds salt-and-pepper noise and logs coordinates
├── traditional_filtering.py                     # Traditional denoising implementations
├── output_images/                               # Clean original images
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
- Add salt-and-pepper noise at specified densities
- Record ground truth noise locations in CSV format

### 2. Traditional Filtering Methods
Implementation of classical denoising techniques:
- **Median Filter**: Non-linear filter effective for impulse noise
- **Adaptive Median Filter**: Variable window size based on local statistics
- **Gaussian Filter**: Linear smoothing filter
- **Bilateral Filter**: Edge-preserving smoothing
- **Non-Local Means**: Patch-based denoising
- **Morphological Filters**: Opening/closing operations

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
python Make_Noise.py
```

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
numpy
opencv-python
scikit-image
matplotlib
```

For deep learning methods:
```
torch
torchvision
tensorflow (alternative)
```

## Results

Results will include:
- Side-by-side visual comparisons (original, noisy, denoised)
- Quantitative metrics table (PSNR/SSIM per method)
- Performance analysis graphs
- Execution time benchmarks
- Discussion of trade-offs

## Future Work

- Extend to other noise types (Gaussian, Poisson, speckle)
- Test on real-world images
- Explore other deep learning architectures (U-Net, RED-Net)
- Implement hybrid approaches
- Real-time denoising applications

## References

- Zhang, K., et al. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
- Buades, A., Coll, B., & Morel, J. M. (2005). "A non-local algorithm for image denoising"
- Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for gray and color images"


## Authors

Rafael Moreno
Alejandro Rubio
Carlos Lopez
Fall 2025 - TTU ECE 4367 Image Processing
