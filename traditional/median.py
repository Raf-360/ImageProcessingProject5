"""
Median filter for salt-and-pepper noise.
"""

import cv2 as cv
import numpy as np


def median_denoise(image: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Median filter (primarily for salt-and-pepper noise).
    
    Pros: Excellent for impulse noise, preserves edges
    Cons: Can blur fine details
    
    Note: Multiple iterations can significantly improve salt-and-pepper noise removal
    without over-blurring like other filters.
    
    Args:
        image: Input noisy image
        kernel_size: Size of the median filter kernel (odd number)
        iterations: Number of times to apply the filter (default: 1)
        
    Returns:
        denoised_image: Filtered image
    """
    result = image.copy()
    for _ in range(iterations):
        result = cv.medianBlur(result, kernel_size)
    return result
