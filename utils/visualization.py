"""
Visualization utilities for denoising results.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def visualize_results(noisy_image: np.ndarray, results_dict: Dict[str, np.ndarray], 
                     ground_truth: Optional[np.ndarray] = None) -> None:
    """
    Visualize multiple denoising results side-by-side.
    
    Args:
        noisy_image: Original noisy image
        results_dict: Dict mapping method_name -> denoised_image
        ground_truth: Optional clean image for comparison
    """
    num_images = len(results_dict) + 1  # +1 for noisy image
    if ground_truth is not None:
        num_images += 1
    
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    idx = 0
    
    # Show ground truth first if available
    if ground_truth is not None:
        axes[idx].imshow(cv.cvtColor(ground_truth, cv.COLOR_BGR2RGB))
        axes[idx].set_title('Ground Truth (Clean)', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # Show noisy image
    axes[idx].imshow(cv.cvtColor(noisy_image, cv.COLOR_BGR2RGB))
    axes[idx].set_title('Noisy Image', fontsize=12, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    # Show denoised results
    for method_name, denoised in results_dict.items():
        axes[idx].imshow(cv.cvtColor(denoised, cv.COLOR_BGR2RGB))
        axes[idx].set_title(f'{method_name.replace("_", " ").title()}', fontsize=12)
        axes[idx].axis('off')
        idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_wiener_filter(fft_magnitude: np.ndarray, wiener_filter: np.ndarray, 
                           noise_variance: float, mysize: int) -> None:
    """
    Visualize Wiener filter's Fourier transform and filter response.
    
    Args:
        fft_magnitude: Log magnitude of FFT
        wiener_filter: Wiener filter response
        noise_variance: Noise variance used
        mysize: Window size used
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot FFT magnitude spectrum (log scale)
    im1 = axes[0].imshow(fft_magnitude, cmap='hot')
    axes[0].set_title('Fourier Transform (Log Magnitude)\nof Noisy Image (Channel 0)', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot Wiener filter
    im2 = axes[1].imshow(wiener_filter, cmap='viridis')
    axes[1].set_title(f'Wiener Filter Response\n(Window Size: {mysize}, Noise Variance: {noise_variance:.2f})', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
