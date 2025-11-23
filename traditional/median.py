"""
Median filter for salt-and-pepper noise.
"""

import cv2 as cv
import numpy as np


def median_denoise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median filter (primarily for salt-and-pepper noise).
    
    Pros: Excellent for impulse noise, preserves edges
    Cons: Can blur fine details
    
    Args:
        image: Input noisy image
        kernel_size: Size of the median filter kernel (odd number)
        
    Returns:
        denoised_image: Filtered image
    """
    return cv.medianBlur(image, kernel_size)
