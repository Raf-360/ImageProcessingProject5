"""
Non-Local Means denoising filter.
"""

import cv2 as cv
import numpy as np


def nlm_denoise(image: np.ndarray, h: float = 10, template_window_size: int = 7, 
               search_window_size: int = 21) -> np.ndarray:
    """
    Non-local means denoising using cv2.fastNlMeansDenoising().
    
    Pros: Excellent for Gaussian noise, state-of-the-art traditional method
    Cons: Very slow, many parameters
    
    Args:
        image: Input noisy image
        h: Filter strength (higher = more denoising)
        template_window_size: Size of patch for comparison (odd)
        search_window_size: Size of search area (odd)
        
    Returns:
        denoised_image: Filtered image
    """
    return cv.fastNlMeansDenoising(image, h=h, templateWindowSize=template_window_size, 
                                   searchWindowSize=search_window_size)
