"""
Bilateral filter for edge-preserving smoothing.
"""

import cv2 as cv
import numpy as np


def bilateral_denoise(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                     sigma_space: float = 75, iterations: int = 1) -> np.ndarray:
    """
    Edge-preserving filter that smooths while keeping boundaries sharp.
    
    This is smarter than Gaussian blur - it considers both spatial distance and color
    similarity when filtering. Slower but preserves important details better.
    
    Note: Multiple iterations can progressively refine the denoising while maintaining
    edge sharpness. 2-3 iterations often work well.
    
    Args:
        image: Noisy input image
        d: Neighborhood size to consider
        sigma_color: How much color difference matters
        sigma_space: How much spatial distance matters
        iterations: Number of times to apply the filter (default: 1)
        
    Returns:
        Filtered image with preserved edges
    """
    result = image.copy()
    for _ in range(iterations):
        result = cv.bilateralFilter(result, d, sigma_color, sigmaSpace=sigma_space)
    return result
