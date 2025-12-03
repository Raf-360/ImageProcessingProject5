"""
Error map visualization utilities.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def generate_error_map(clean: np.ndarray, denoised: np.ndarray, 
                       error_type: str = 'absolute') -> np.ndarray:
    """
    Generate an error map showing differences between clean and denoised images.
    
    Args:
        clean: Ground truth clean image
        denoised: Denoised image to compare
        error_type: Type of error to compute ('absolute', 'squared', 'normalized')
        
    Returns:
        error_map: Grayscale error map
    """
    # Convert to grayscale if color
    if len(clean.shape) == 3:
        clean_gray = cv.cvtColor(clean, cv.COLOR_BGR2GRAY)
        denoised_gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)
    else:
        clean_gray = clean
        denoised_gray = denoised
    
    # Compute error
    if error_type == 'absolute':
        error = np.abs(clean_gray.astype(float) - denoised_gray.astype(float))
    elif error_type == 'squared':
        error = (clean_gray.astype(float) - denoised_gray.astype(float)) ** 2
    elif error_type == 'normalized':
        diff = np.abs(clean_gray.astype(float) - denoised_gray.astype(float))
        error = diff / (clean_gray.astype(float) + 1e-8)  # Avoid division by zero
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    # Normalize to 0-255 range
    error_normalized = ((error - error.min()) / (error.max() - error.min() + 1e-8) * 255).astype(np.uint8)
    
    return error_normalized


def visualize_error_map(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
                        method_name: str = "Method", save_path: Optional[str] = None,
                        show: bool = True):
    """
    Visualize error maps comparing noisy and denoised images to ground truth.
    
    Args:
        clean: Ground truth clean image
        noisy: Noisy input image
        denoised: Denoised output image
        method_name: Name of the denoising method
        save_path: Optional path to save the visualization
    """
    # Generate error maps
    error_noisy = generate_error_map(clean, noisy, 'absolute')
    error_denoised = generate_error_map(clean, denoised, 'absolute')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    if len(clean.shape) == 3:
        axes[0, 0].imshow(cv.cvtColor(clean, cv.COLOR_BGR2RGB))
        axes[0, 1].imshow(cv.cvtColor(noisy, cv.COLOR_BGR2RGB))
        axes[0, 2].imshow(cv.cvtColor(denoised, cv.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(clean, cmap='gray')
        axes[0, 1].imshow(noisy, cmap='gray')
        axes[0, 2].imshow(denoised, cmap='gray')
    
    axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Noisy Input', fontsize=12, fontweight='bold')
    axes[0, 2].set_title(f'{method_name} Output', fontsize=12, fontweight='bold')
    
    for ax in axes[0]:
        ax.axis('off')
    
    # Row 2: Error maps
    im1 = axes[1, 0].imshow(error_noisy, cmap='hot')
    axes[1, 0].set_title('Error: Noisy vs Ground Truth', fontsize=11)
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(error_denoised, cmap='hot')
    axes[1, 1].set_title(f'Error: {method_name} vs Ground Truth', fontsize=11)
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Improvement map (difference between error maps)
    improvement = error_noisy.astype(float) - error_denoised.astype(float)
    im3 = axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-50, vmax=50)
    axes[1, 2].set_title('Improvement Map\n(Green=Better, Red=Worse)', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    for ax in axes[1]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error map visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def visualize_multi_method_errors(clean: np.ndarray, noisy: np.ndarray, 
                                  results_dict: dict, save_path: Optional[str] = None,
                                  show: bool = True):
    """
    Compare error maps for multiple denoising methods.
    
    Args:
        clean: Ground truth clean image
        noisy: Noisy input image
        results_dict: Dict mapping method_name -> denoised_image
        save_path: Optional path to save the visualization
        show: Whether to display the plot (default: True)
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    if n_methods == 0:
        axes = axes.reshape(2, -1)
    
    # Column 0: Ground truth and its placeholder
    if len(clean.shape) == 3:
        axes[0, 0].imshow(cv.cvtColor(clean, cv.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Noisy error map
    error_noisy = generate_error_map(clean, noisy, 'absolute')
    im0 = axes[1, 0].imshow(error_noisy, cmap='hot')
    axes[1, 0].set_title('Noisy Error', fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Denoised results and error maps
    for idx, (method_name, denoised) in enumerate(results_dict.items(), start=1):
        # Denoised image
        if len(denoised.shape) == 3:
            axes[0, idx].imshow(cv.cvtColor(denoised, cv.COLOR_BGR2RGB))
        else:
            axes[0, idx].imshow(denoised, cmap='gray')
        axes[0, idx].set_title(f'{method_name.upper()}', fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Error map
        error_map = generate_error_map(clean, denoised, 'absolute')
        im = axes[1, idx].imshow(error_map, cmap='hot')
        axes[1, idx].set_title(f'{method_name.upper()} Error', fontsize=10)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-method error map to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def visualize_noise_distribution(noisy: np.ndarray, clean: np.ndarray,
                                save_path: Optional[str] = None, show: bool = True):
    """
    Visualize the noise distribution in the image.
    
    Args:
        noisy: Noisy image
        clean: Clean ground truth image
        save_path: Optional path to save the plot
        show: Whether to display the plot (default: True)
    """
    # Calculate noise
    noise = noisy.astype(float) - clean.astype(float)
    
    # Convert to grayscale for analysis if color
    if len(noise.shape) == 3:
        noise_gray = cv.cvtColor(noise.astype(np.uint8), cv.COLOR_BGR2GRAY).astype(float)
    else:
        noise_gray = noise
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Noise histogram
    axes[0, 0].hist(noise_gray.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Noise Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Noise Distribution Histogram', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].legend()
    
    # 2. Noise visualization
    noise_display = ((noise_gray - noise_gray.min()) / (noise_gray.max() - noise_gray.min() + 1e-8) * 255).astype(np.uint8)
    im1 = axes[0, 1].imshow(noise_display, cmap='RdBu_r')
    axes[0, 1].set_title('Noise Spatial Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Noise magnitude (absolute)
    noise_magnitude = np.abs(noise_gray)
    im2 = axes[1, 0].imshow(noise_magnitude, cmap='hot')
    axes[1, 0].set_title('Noise Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. Statistics box
    axes[1, 1].axis('off')
    stats_text = f"""
    Noise Statistics:
    
    Mean: {noise_gray.mean():.4f}
    Std Dev: {noise_gray.std():.4f}
    Min: {noise_gray.min():.4f}
    Max: {noise_gray.max():.4f}
    
    Median: {np.median(noise_gray):.4f}
    MAD: {np.median(np.abs(noise_gray - np.median(noise_gray))):.4f}
    
    SNR: {10 * np.log10(clean.astype(float).var() / (noise_gray.var() + 1e-8)):.2f} dB
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_title('Noise Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise distribution to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
