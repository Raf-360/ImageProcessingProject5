"""
Non-Local Means denoising filter.
"""

import cv2 as cv
import numpy as np


def nlm_denoise(image: np.ndarray, h: float = 10, template_window_size: int = 7, 
               search_window_size: int = 21) -> np.ndarray:
    """
    Non-Local Means - one of the best traditional denoising methods out there.
    
    Searches for similar patches across the entire image and averages them.
    Super effective but pretty slow. Worth it for high-quality results though.
    
    Args:
        image: Noisy input image
        h: Filtering strength (higher = more aggressive denoising)
        template_window_size: Patch size for comparing similarity
        search_window_size: How far to search for similar patches
        
    Returns:
        Denoised image
    """
    return cv.fastNlMeansDenoising(image, h=h, templateWindowSize=template_window_size, 
                                   searchWindowSize=search_window_size)
