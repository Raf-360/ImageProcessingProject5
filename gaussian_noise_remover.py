
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math
import time
import sys

LAPLACIAN_KERNAL_ENERGY = 72.0


class TraditionalNoiseRemover:
    """
    A comprehensive denoising toolkit for removing noise
    from images using multiple traditional filtering techniques.
    """
    
    def __init__(self, image_path: Path, ground_truth: Path): 
        """
        Initialize the denoiser with a noisy image.
        
        Args:
            image_path: Noisey Image File Path
            ground_truth:  Original/Clean images
        """
        
        _images, _ground_truths = self._load_image(image_path, ground_truth)
        
        self.images = _images
        self.ground_truths = _ground_truths# images to clean
       
        
        self._denoised_cache: Dict[str, np.ndarray] = {}
        self._metrics_cache: Dict[str, Dict[str, float]] = {}
        
    
    def _load_image(self, noisey_images_path: Path = None, ground_truths_image_path: Optional[Path] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load image from file path or validate numpy array.
        
        Args:
            noisey_images_path: File path containing noisy images 
            ground_truths_image_path: File path containing clean/ground truth images
            
        Returns:
            Tuple of (loaded_noisey_images, loaded_gt_images)
        """
        
        # Load noisy images
        loaded_noisey_images = []
        if noisey_images_path is not None:
            noisey_images = sorted(list(noisey_images_path.rglob("*.png")))
            for image_path in noisey_images:
                img = cv.imread(str(image_path))
                if img is not None:
                    loaded_noisey_images.append(img)
        
        # Load ground truth images
        loaded_gt_images = []
        if ground_truths_image_path is not None:
            ground_truth_images = sorted(list(ground_truths_image_path.rglob("*.png")))
            for image_path in ground_truth_images:
                img = cv.imread(str(image_path))
                if img is not None:
                    loaded_gt_images.append(img)
        
        return loaded_noisey_images, loaded_gt_images 
            
            
        
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to float32 [0, 1] range for processing.
        
        Args:
            image: Input image
            
        Returns:
            normalized: Normalized image

        """
        
        image = image.astype(np.float32) / 255.0
        return image
    
    def _denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to uint8 [0, 255] range.
        
        Args:
            image: Normalized image
            
        Returns:
            denormalized: Image in uint8 format
        """
        image = np.clip(image*255.0, 0, 255.0).astype(np.uint8)
        return image
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate the noise level (sigma) in the image(assuming no ground truth).
        
        Returns:
            noise level
        """
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Force consistent kernel
        lap = cv.Laplacian(gray, cv.CV_64F, ksize=1)

        # Only use smooth regions for MAD (edge mask)
        edges = cv.Canny(gray, 80, 160)
        mask = (edges == 0)

        if np.sum(mask) < 50:
            # fallback if mask too small
            data = lap
        else:
            data = lap[mask]

        mad = np.median(np.abs(data - np.median(data)))

        # kernel energy for 3x3 4-connected Laplacian (ksize=1)
        LAPLACIAN_KERNEL_ENERGY = 20

        sigma_lap = 1.4826 * mad
        sigma_n = sigma_lap / math.sqrt(LAPLACIAN_KERNEL_ENERGY)

        print(f"M.A.D.:          {mad:.2f}")
        print(f"Estimated Sigma: {sigma_n:.2f}\n")

        return sigma_n
    
    
    # ========== Core Denoising Methods ==========
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                     sigma: float = 1.0) -> np.ndarray:
        """
        Simple Gaussian blur using cv2.GaussianBlur().
        
        Pros: Fast, simple baseline
        Cons: Blurs edges, not adaptive
        
        Args:
            kernel_size: Tuple (width, height), must be odd
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            denoised_image: Filtered image (same shape as input)
        """
        blurred_img = cv.GaussianBlur(image, kernel_size, 0)
        return blurred_img
    
    def bilateral_filter(self, d: int = 9, sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Edge-preserving smoothing using cv2.bilateralFilter().
        
        Pros: Preserves edges while smoothing
        Cons: Slower than Gaussian, parameter-sensitive
        
        Args:
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            denoised_image: Filtered image
        """
        
    
    def non_local_means(self, h: float = 10, template_window_size: int = 7, 
                       search_window_size: int = 21) -> np.ndarray:
        """
        Non-local means denoising using cv2.fastNlMeansDenoisingColored().
        
        Pros: Excellent for Gaussian noise, state-of-the-art traditional method
        Cons: Very slow, many parameters
        
        Args:
            h: Filter strength (higher = more denoising)
            template_window_size: Size of patch for comparison (odd)
            search_window_size: Size of search area (odd)
            
        Returns:
            denoised_image: Filtered image
        """
        pass
    
    def wiener_filter(self, noise_variance: Optional[float] = None) -> np.ndarray:
        """
        Optimal linear filter for Gaussian noise using Wiener filtering.
        
        Note: Requires frequency domain processing
        Can auto-estimate noise_variance if None
        
        Args:
            noise_variance: Estimated noise power (auto-detect if None)
            
        Returns:
            denoised_image: Filtered image
        """
        pass
    
    def median_filter(self, kernel_size: int = 5) -> np.ndarray:
        """
        Median filter (primarily for salt-and-pepper, but included for comparison).
        
        Args:
            kernel_size: Size of the median filter kernel (odd number)
            
        Returns:
            denoised_image: Filtered image
        """
        pass

    
    def denoise_all_methods(self, params_dict: Optional[Dict[str, Dict]] = None) -> Dict[str, np.ndarray]:
        """
        Run all denoising methods with specified parameters.
        
        Args:
            params_dict: Dictionary mapping method names to their parameters
                Example: {
                    'gaussian_blur': {'kernel_size': (5,5), 'sigma': 1.0},
                    'bilateral': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                    'nlm': {'h': 10, 'template_window_size': 7, 'search_window_size': 21}
                }
                If None, use default parameters
                
        Returns:
            results: Dict mapping method_name -> denoised_image
        """
        pass
    
    def auto_tune(self, method_name: str, param_ranges: Dict[str, List], 
                  metric: str = 'psnr') -> Tuple[Dict, float]:
        """
        Grid search to find optimal parameters for a method.
        
        Requires ground_truth to be set!
        
        Args:
            method_name: 'gaussian_blur', 'bilateral', 'nlm', etc.
            param_ranges: Dict of parameter ranges to search
                Example: {'d': [5, 7, 9], 'sigma_color': [50, 75, 100]}
            metric: 'psnr' or 'ssim'
            
        Returns:
            best_params: Dict of optimal parameters
            best_score: Best PSNR/SSIM achieved
        """
        pass
    
    # ========== Evaluation Methods ==========
    
    def calculate_psnr(self, denoised_image: np.ndarray, 
                      ground_truth: Optional[np.ndarray] = None) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        PSNR = 10 * log10(MAX^2 / MSE)
        Higher is better (typically 20-40 dB range)
        
        Args:
            denoised_image: Result from denoising
            ground_truth: Clean reference (uses self._ground_truth if None)
            
        Returns:
            psnr_value: PSNR in dB
        """
        pass
    
    def calculate_ssim(self, denoised_image: np.ndarray, 
                      ground_truth: Optional[np.ndarray] = None) -> float:
        """
        Calculate Structural Similarity Index.
        
        Range: [-1, 1], where 1 = identical
        Considers luminance, contrast, structure
        
        Args:
            denoised_image: Result from denoising
            ground_truth: Clean reference
            
        Returns:
            ssim_value: SSIM score
        """
        pass
    
    def compare_methods(self, ground_truth: Optional[np.ndarray] = None, 
                       methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run multiple methods and generate comparison report.
        
        Args:
            ground_truth: Clean image for metrics
            methods: List of method names (None = all methods)
            
        Returns:
            comparison_df: Pandas DataFrame with:
                - Method name
                - PSNR
                - SSIM
                - Processing time
                - Parameters used
        """
        pass
    
    # ========== Utility Methods ==========
    
    def save_result(self, denoised_image: np.ndarray, output_path: Union[str, Path]) -> None:
        """
        Save denoised image to file.
        
        Args:
            denoised_image: Image to save
            output_path: Output file path
        """
        pass
    
    def visualize_results(self, results_dict: Dict[str, np.ndarray]) -> None:
        """
        Visualize multiple denoising results side-by-side.
        
        Args:
            results_dict: Dict mapping method_name -> denoised_image
        """
        pass
    
    def get_denoised_image(self, method_name: str) -> Optional[np.ndarray]:
        """
        Retrieve cached denoised image from previous run.
        
        Args:
            method_name: Name of the denoising method
            
        Returns:
            denoised_image: Cached result or None if not found
        """
        pass
    
    @property
    def original_image(self) -> np.ndarray:
        """Get the original noisy image."""
        return self.images.copy()
    
    @property
    def ground_truth_image(self) -> Optional[np.ndarray]:
        """Get the ground truth clean image (if available)."""
        return self.ground_truths.copy() if self.ground_truths is not None else None
    
    @property
    def available_methods(self) -> List[str]:
        """Get list of available denoising methods."""
        return ['gaussian_blur', 'bilateral', 'non_local_means', 'wiener', 'median']
    
    
    

def main():
    
    # Set up paths
    noisy_path = Path("gaussian_noise_25_sigma")
    clean_path = Path("clean_images")
    
    # first N images to be processed(so we don't ahve to look at them all)
    num_images = 5
    
    # Check if directories exist
    if not noisy_path.exists():
        print(f"Error: Directory '{noisy_path}' not found")
        sys.exit(1)
    
    if not clean_path.exists():
        print(f"Error: Directory '{clean_path}' not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Loading Images")
    print("=" * 60)
    
    try:
        # Initialize the denoiser
        print("\n1. Initializing denoiser...")
        denoiser = TraditionalNoiseRemover(
            image_path=noisy_path,
            ground_truth=clean_path
        )
        
        # Check loaded images
        print(f"✓ Loaded {len(denoiser.images)} noisy images")
        print(f"✓ Loaded {len(denoiser.ground_truths)} ground truth images\n")
        
        if len(denoiser.images) == 0:
            print("Error: No noisy images loaded!")
            sys.exit(1)
        
        print("===== Starting Analysis + Noise Remover =====\n")
        
        print("============ Initial Image Stuff ============")
        for i in range(num_images):
            noisey_img = denoiser.images[i]
            ground_truth_img = denoiser.ground_truths[i]
            print(f"Noisey image size:              {noisey_img.shape}")
            print(f"Ground Truth Image Dimensions:  {ground_truth_img.shape}")
            print(f"{'-'*25}")
            print(f"Noisey Image dtype:             {noisey_img.dtype}")
            print(f"Ground Truth Image dtype:       {ground_truth_img.dtype}")
            
        
        print("=========== Estimated Noise Level ===========")
        for i in range(num_images):
            print(f"Image {i+1}")
            img = denoiser.images[i]
            noise_est = denoiser._estimate_noise_level(img)
        
        
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    
    
    
if __name__ == "__main__":
    main()