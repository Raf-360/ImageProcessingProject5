"""
Gaussian blur denoising filter.
"""

import cv2 as cv
import numpy as np
from typing import Tuple


def gaussian_denoise(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                    sigma: float = 1.0, iterations: int = 1) -> np.ndarray:
    """
    Basic Gaussian blur - the simplest approach to noise reduction.
    
    Works well as a baseline but tends to blur edges. Fast though!
    
    Note: Multiple iterations with smaller sigma can provide more controlled smoothing
    than a single pass with large sigma.
    
    Args:
        image: Noisy input image
        kernel_size: Filter window size (width, height) - must be odd numbers
        sigma: Controls blur strength
        iterations: Number of times to apply the filter (default: 1)
        
    Returns:
        Smoothed image
    """
    result = image.copy()
    for _ in range(iterations):
        result = cv.GaussianBlur(result, kernel_size, sigma)
    return result
