"""
Image quality metrics for denoising evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(denoised_image: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio between denoised and ground truth images.
    
    PSNR uses a logarithmic scale (dB) to measure reconstruction quality.
    Higher values mean better quality - typically you'll see 20-40 dB for decent results.
    
    Args:
        denoised_image: The result after denoising
        ground_truth: Original clean image for comparison
        
    Returns:
        PSNR value in decibels
    """
    return peak_signal_noise_ratio(ground_truth, denoised_image, data_range=255)


def calculate_ssim(denoised_image: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Measures structural similarity between images - more sophisticated than just pixel differences.
    
    Considers luminance, contrast, and structural patterns. Values closer to 1 indicate
    better preservation of image structure during denoising.
    
    Args:
        denoised_image: The result after denoising
        ground_truth: Original clean image for comparison
        
    Returns:
        Similarity score between -1 and 1
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
