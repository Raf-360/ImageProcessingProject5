"""
Noise estimation utilities.
"""

import cv2 as cv
import numpy as np


LAPLACIAN_KERNEL_ENERGY = 72.0


def estimate_noise_level(image: np.ndarray) -> float:
    """
    Tries to estimate how much noise is in the image using the MAD method.
    
    Uses a Laplacian filter to isolate high-frequency components (which are mostly noise),
    then computes the median absolute deviation to get a robust noise estimate.
    
    Args:
        image: Input image (works with BGR or grayscale)
        
    Returns:
        Estimated noise level (sigma)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Laplacian filter
    laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    
    # Compute MAD (Median Absolute Deviation)
    data = laplacian.flatten()
    mad = np.median(np.abs(data - np.median(data)))
    
    # Estimate sigma using MAD
    sigma = mad / 0.6745
    
    # Normalize by kernel energy
    sigma_normalized = sigma / np.sqrt(LAPLACIAN_KERNEL_ENERGY)
    
    print(f"  Estimated noise level (σ): {sigma_normalized:.4f}")
    
    return sigma_normalized
