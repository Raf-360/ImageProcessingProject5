"""
Wiener filter for optimal linear denoising.
"""

import numpy as np
from scipy.signal import wiener
from typing import Optional, Dict


def wiener_denoise(image: np.ndarray, mysize: int = 5, 
                  noise_variance: Optional[float] = None) -> tuple[np.ndarray, Optional[Dict]]:
    """
    Optimal linear filter for Gaussian noise using Wiener filtering.
    Uses scipy.signal.wiener for efficient spatial-domain filtering.
    
    Args:
        image: Input noisy image
        mysize: Size of the Wiener filter window (e.g., 3, 5, 7)
        noise_variance: Estimated noise power (auto-detect if None)
        
    Returns:
        Tuple of (denoised_image, viz_data)
        viz_data contains FFT and filter response for visualization
    """
    # Normalize to [0, 1] range
    image_norm = image.astype(np.float32) / 255.0
    
    # Process each channel separately
    denoised = np.zeros_like(image_norm, dtype=np.float32)
    
    # Auto-estimate noise variance if not provided
    if noise_variance is None:
        from utils.noise_estimation import estimate_noise_level
        sigma = estimate_noise_level(image)
        noise_variance = sigma ** 2
    
    # Storage for visualization (use first channel)
    viz_data = {}
    
    for channel in range(image.shape[2]):
        # Get channel data
        img_channel = image_norm[:, :, channel]
        
        # Apply scipy Wiener filter
        filtered = wiener(img_channel, mysize=mysize, noise=noise_variance)
        
        # Store visualization data for first channel (FFT of input and filter response)
        if channel == 0:
            # Compute FFT for visualization
            f_transform = np.fft.fft2(img_channel)
            f_shift = np.fft.fftshift(f_transform)
            
            # Compute FFT of filtered result to show filter response
            f_transform_filtered = np.fft.fft2(filtered)
            f_shift_filtered = np.fft.fftshift(f_transform_filtered)
            
            # Compute approximate Wiener filter response in frequency domain
            epsilon = 1e-10  # Avoid division by zero
            wiener_response = np.abs(f_shift_filtered) / (np.abs(f_shift) + epsilon)
            
            viz_data['fft_magnitude'] = np.log(np.abs(f_shift) + 1)
            viz_data['wiener_filter'] = wiener_response
            viz_data['noise_variance'] = noise_variance
            viz_data['mysize'] = mysize
        
        # Store filtered channel
        denoised[:, :, channel] = np.clip(filtered, 0, 1)
    
    # Convert back to uint8 [0, 255] format
    denoised = (denoised * 255.0).astype(np.uint8)
    
    return denoised, viz_data
