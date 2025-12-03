"""Traditional denoising methods."""

from .gaussian import gaussian_denoise
from .median import median_denoise
from .bilateral import bilateral_denoise
from .nlm import nlm_denoise
from .wiener import wiener_denoise

__all__ = [
    'gaussian_denoise',
    'median_denoise',
    'bilateral_denoise',
    'nlm_denoise',
    'wiener_denoise'
]
