"""
Gaussian blur denoising filter.
"""

import cv2 as cv
import numpy as np
from typing import Tuple


def gaussian_denoise(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                    sigma: float = 1.0) -> np.ndarray:
    """
    Basic Gaussian blur - the simplest approach to noise reduction.
    
    Works well as a baseline but tends to blur edges. Fast though!
    
    Args:
        image: Noisy input image
        kernel_size: Filter window size (width, height) - must be odd numbers
        sigma: Controls blur strength
        
    Returns:
        Smoothed image
    """
    return cv.GaussianBlur(image, kernel_size, sigma)
