"""
Image quality metrics for denoising evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(denoised_image: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    Higher is better (typically 20-40 dB range)
    
    Args:
        denoised_image: Result from denoising
        ground_truth: Clean reference
        
    Returns:
        psnr_value: PSNR in dB
    """
    return peak_signal_noise_ratio(ground_truth, denoised_image, data_range=255)


def calculate_ssim(denoised_image: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index.
    
    SSIM compares luminance, contrast, and structure.
    Range: [-1, 1], where 1 means identical images.
    
    Args:
        denoised_image: Result from denoising
        ground_truth: Clean reference
        
    Returns:
        ssim_value: Similarity index
    """
    return structural_similarity(ground_truth, denoised_image, channel_axis=2, data_range=255)


def calculate_mse(denoised_image: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        denoised_image: Result from denoising
        ground_truth: Clean reference
        
    Returns:
        mse_value: Mean squared error
    """
    return np.mean((denoised_image.astype(float) - ground_truth.astype(float)) ** 2)
