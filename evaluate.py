"""
Evaluation script for batch testing and metrics generation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

from utils.image_io import load_image_pair
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse
from traditional.gaussian import gaussian_denoise
from traditional.median import median_denoise
from traditional.bilateral import bilateral_denoise
from traditional.nlm import nlm_denoise
from traditional.wiener import wiener_denoise


def evaluate_method(method_name: str, noisy_images: List, clean_images: List) -> Dict:
    """
    Evaluate a single denoising method on all images.
    
    Args:
        method_name: Name of the method
        noisy_images: List of noisy images
        clean_images: List of clean images
        
    Returns:
        Dictionary with aggregated metrics
    """
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    
    for noisy, clean in zip(noisy_images, clean_images):
        # Apply denoising
        if method_name == 'gaussian':
            denoised, _ = gaussian_denoise(noisy), None
        elif method_name == 'median':
            denoised, _ = median_denoise(noisy), None
        elif method_name == 'bilateral':
            denoised, _ = bilateral_denoise(noisy), None
        elif method_name == 'nlm':
            denoised, _ = nlm_denoise(noisy), None
        elif method_name == 'wiener':
            denoised, _ = wiener_denoise(noisy)
        else:
            continue
        
        # Calculate metrics
        psnr_scores.append(calculate_psnr(denoised, clean))
        ssim_scores.append(calculate_ssim(denoised, clean))
        mse_scores.append(calculate_mse(denoised, clean))
    
    return {
        'Method': method_name.upper(),
        'Mean PSNR': np.mean(psnr_scores),
        'Std PSNR': np.std(psnr_scores),
        'Mean SSIM': np.mean(ssim_scores),
        'Std SSIM': np.std(ssim_scores),
        'Mean MSE': np.mean(mse_scores),
        'Std MSE': np.std(mse_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of denoising methods")
    parser.add_argument("-n", "--noisy", type=str, required=True, help="Noisy images directory")
    parser.add_argument("-c", "--clean", type=str, required=True, help="Clean images directory")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file for results")
    parser.add_argument("--methods", nargs='+', default=['gaussian', 'median', 'bilateral', 'nlm', 'wiener'],
                       help="Methods to evaluate")
    
    args = parser.parse_args()
    
    noisy_path = Path(args.noisy)
    clean_path = Path(args.clean)
    
    if not noisy_path.exists() or not clean_path.exists():
        print("❌ Error: Input directories not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Batch Evaluation")
    print("=" * 60)
    
    # Load images
    print("\n📂 Loading images...")
    noisy_images, clean_images = load_image_pair(noisy_path, clean_path)
    print(f"✓ Loaded {len(noisy_images)} image pairs\n")
    
    if len(noisy_images) != len(clean_images):
        print("⚠️  Warning: Number of noisy and clean images don't match")
    
    # Evaluate each method
    results = []
    for method in args.methods:
        print(f"Evaluating {method}...")
        result = evaluate_method(method, noisy_images, clean_images)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
