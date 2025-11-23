"""
Bilateral filter for edge-preserving smoothing.
"""

import cv2 as cv
import numpy as np


def bilateral_denoise(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                     sigma_space: float = 75) -> np.ndarray:
    """
    Edge-preserving filter that smooths while keeping boundaries sharp.
    
    This is smarter than Gaussian blur - it considers both spatial distance and color
    similarity when filtering. Slower but preserves important details better.
    
    Args:
        image: Noisy input image
        d: Neighborhood size to consider
        sigma_color: How much color difference matters
        sigma_space: How much spatial distance matters
        
    Returns:
        Filtered image with preserved edges
    """
    return cv.bilateralFilter(image, d, sigma_color, sigmaSpace=sigma_space)
