"""
Bilateral filter for edge-preserving smoothing.
"""

import cv2 as cv
import numpy as np


def bilateral_denoise(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                     sigma_space: float = 75) -> np.ndarray:
    """
    Edge-preserving smoothing using cv2.bilateralFilter().
    
    Pros: Preserves edges while smoothing
    Cons: Slower than Gaussian, parameter-sensitive
    
    Args:
        image: Input noisy image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        denoised_image: Filtered image
    """
    return cv.bilateralFilter(image, d, sigma_color, sigmaSpace=sigma_space)
