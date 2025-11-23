"""
Gaussian blur denoising filter.
"""

import cv2 as cv
import numpy as np
from typing import Tuple


def gaussian_denoise(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                    sigma: float = 1.0) -> np.ndarray:
    """
    Simple Gaussian blur using cv2.GaussianBlur().
    
    Pros: Fast, simple baseline
    Cons: Blurs edges, not adaptive
    
    Args:
        image: Input noisy image
        kernel_size: Tuple (width, height), must be odd
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        denoised_image: Filtered image
    """
    return cv.GaussianBlur(image, kernel_size, sigma)
