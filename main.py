"""
Single-Image All-Methods Denoising Tool

Applies all implemented denoising techniques to a single noisy image.
Supports optional ground truth for quality metrics (PSNR/SSIM).

Usage:
    python main.py noisy_image.png                    # Without ground truth
    python main.py noisy_image.png clean_image.png    # With ground truth for metrics
"""
import sys
import numpy as np
import cv2 as cv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Essential imports
from utils.metrics import calculate_psnr, calculate_ssim


def print_usage():
    """Print usage instructions and exit."""
    print("Usage:")
    print("  python main.py <noisy_image_path> [ground_truth_path]")
    print("\nExamples:")
    print("  python main.py noisy.png")
    print("  python main.py noisy.png clean.png")
    print("\nArguments:")
    print("  noisy_image_path     - Path to noisy input image (required)")
    print("  ground_truth_path    - Path to clean reference image (optional, enables metrics)")
    sys.exit(1)


def load_images(noisy_path: str, ground_truth_path: Optional[str] = None):
    """Load noisy image and optional ground truth, convert to grayscale."""
    noisy_img = cv.imread(noisy_path)
    if noisy_img is None:
        print(f"Error: Could not load image at {noisy_path}")
        sys.exit(1)
    
    if len(noisy_img.shape) == 3:
        noisy_gray = cv.cvtColor(noisy_img, cv.COLOR_BGR2GRAY)
    else:
        noisy_gray = noisy_img
    
    ground_truth_gray = None
    if ground_truth_path:
        ground_truth_img = cv.imread(ground_truth_path)
        if ground_truth_img is None:
            print(f"Warning: Could not load ground truth at {ground_truth_path}")
            print("Continuing without metrics...")
        else:
            if len(ground_truth_img.shape) == 3:
                ground_truth_gray = cv.cvtColor(ground_truth_img, cv.COLOR_BGR2GRAY)
            else:
                ground_truth_gray = ground_truth_img
    
    filename_stem = Path(noisy_path).stem
    
    return noisy_gray, ground_truth_gray, filename_stem


def apply_all_methods(noisy_gray: np.ndarray, ground_truth_gray: Optional[np.ndarray] = None) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Apply all denoising methods and display metrics."""
    print("\nApplying denoising methods...")
    traditional_results = {}
    dncnn_results = {}
    
    from traditional.gaussian import gaussian_denoise
    from traditional.median import median_denoise
    from traditional.bilateral import bilateral_denoise
    from traditional.nlm import nlm_denoise
    from traditional.wiener import wiener_denoise
    
    print("[1/8] Gaussian filter...")
    traditional_results['Gaussian'] = gaussian_denoise(noisy_gray, kernel_size=(5, 5), sigma=1.0)
    
    print("[2/8] Median filter...")
    traditional_results['Median'] = median_denoise(noisy_gray, kernel_size=5)
    
    print("[3/8] Bilateral filter...")
    traditional_results['Bilateral'] = bilateral_denoise(noisy_gray, d=9, sigma_color=75, sigma_space=75)
    
    print("[4/8] NLM filter...")
    traditional_results['NLM'] = nlm_denoise(noisy_gray, h=10)
    
    print("[5/8] Wiener filter...")
    wiener_result = wiener_denoise(noisy_gray)
    traditional_results['Wiener'] = wiener_result[0] if isinstance(wiener_result, tuple) else wiener_result
    
    import torch
    from deep.inference import DnCNNDenoiser
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("[6/8] DnCNN (Gaussian)...")
    gaussian_denoiser = DnCNNDenoiser(checkpoint_path='models/gaussian_dncnn.pth', device=device)
    gaussian_result = gaussian_denoiser.denoise(noisy_gray)
    if len(gaussian_result.shape) == 3:
        gaussian_result = cv.cvtColor(gaussian_result, cv.COLOR_BGR2GRAY)
    dncnn_results['DnCNN (Gaussian)'] = gaussian_result
    
    print("[7/8] DnCNN (Salt & Pepper)...")
    sp_denoiser = DnCNNDenoiser(checkpoint_path='models/salt_pepper_dncnn.pth', device=device)
    sp_result = sp_denoiser.denoise(noisy_gray)
    if len(sp_result.shape) == 3:
        sp_result = cv.cvtColor(sp_result, cv.COLOR_BGR2GRAY)
    dncnn_results['DnCNN (Salt & Pepper)'] = sp_result
    
    print("[8/8] DnCNN (Motion Blur)...")
    mb_denoiser = DnCNNDenoiser(checkpoint_path='models/motionblur_dncnn.pth', device=device)
    mb_result = mb_denoiser.denoise(noisy_gray)
    if len(mb_result.shape) == 3:
        mb_result = cv.cvtColor(mb_result, cv.COLOR_BGR2GRAY)
    dncnn_results['DnCNN (Motion Blur)'] = mb_result
    
    print("\nAll methods applied successfully.")
    
    print("\nQuality Metrics:")
    print("-" * 70)
    all_results = {**traditional_results, **dncnn_results}
    for method, img in all_results.items():
        category = "[DnCNN]" if "DnCNN" in method else "[Trad] "
        if ground_truth_gray is not None:
            psnr = calculate_psnr(ground_truth_gray, img)
            ssim = calculate_ssim(ground_truth_gray, img)
            print(f"{category} {method:25s} - PSNR: {psnr:6.2f} dB, SSIM: {ssim:.3f}")
        else:
            psnr = calculate_psnr(noisy_gray, img)
            print(f"{category} {method:25s} - PSNR: {psnr:6.2f} dB")
    print("-" * 70)
    
    return traditional_results, dncnn_results


def visualize_all(noisy_gray: np.ndarray,
                 traditional_results: Dict[str, np.ndarray],
                 dncnn_results: Dict[str, np.ndarray],
                 ground_truth_gray: Optional[np.ndarray],
                 filename: str):
    """Display visualization sequence of all denoising results."""
    import matplotlib.pyplot as plt
    
    has_ground_truth = ground_truth_gray is not None
    
    print("\n[1/6] Displaying input images...")
    if has_ground_truth:
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(noisy_gray, cmap='gray')
        axes[0].set_title('Noisy Input (Grayscale)', fontweight='bold', fontsize=14)
        noisy_psnr = calculate_psnr(ground_truth_gray, noisy_gray)
        noisy_ssim = calculate_ssim(ground_truth_gray, noisy_gray)
        axes[0].text(0.5, -0.1, f'PSNR: {noisy_psnr:.2f} dB\nSSIM: {noisy_ssim:.3f}',
                    ha='center', transform=axes[0].transAxes, fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(ground_truth_gray, cmap='gray')
        axes[1].set_title('Clean Ground Truth (Grayscale)', fontweight='bold', fontsize=14)
        axes[1].text(0.5, -0.1, 'Reference', ha='center', transform=axes[1].transAxes, fontsize=11)
        axes[1].axis('off')
        
        fig1.suptitle(f'Input Images (Grayscale): {filename}', fontsize=16, fontweight='bold')
    else:
        fig1, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(noisy_gray, cmap='gray')
        ax.set_title('Noisy Input (Grayscale)', fontweight='bold', fontsize=14)
        ax.axis('off')
        fig1.suptitle(f'Input Image (Grayscale): {filename}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("[2/6] Displaying input histograms...")
    if has_ground_truth:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(noisy_gray.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
        axes[0].set_title('Noisy Input Histogram', fontweight='bold')
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(ground_truth_gray.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
        axes[1].set_title('Clean (Ground Truth) Histogram', fontweight='bold')
        axes[1].set_xlabel('Pixel Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(alpha=0.3)
        
        fig2.suptitle(f'Input Histograms: {filename}', fontsize=16, fontweight='bold')
    else:
        fig2, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(noisy_gray.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
        ax.set_title('Noisy Input Histogram', fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        fig2.suptitle(f'Input Histogram: {filename}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("[3/6] Displaying traditional filters...")
    n_trad = len(traditional_results)
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (method, img) in enumerate(traditional_results.items()):
        axes[i].imshow(img, cmap='gray')
        
        psnr = calculate_psnr(ground_truth_gray if has_ground_truth else noisy_gray, img)
        if has_ground_truth:
            ssim = calculate_ssim(ground_truth_gray, img)
            title = f'{method}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}'
        else:
            title = f'{method}\nPSNR: {psnr:.2f} dB'
        
        axes[i].set_title(title, fontweight='bold', fontsize=12)
        axes[i].axis('off')
    
    # Hide unused subplot
    if n_trad < 6:
        axes[5].axis('off')
    
    fig2.suptitle('Traditional Filtering Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("[4/6] Displaying histograms of traditional filters...")
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (method, img) in enumerate(traditional_results.items()):
        axes[i].hist(img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        axes[i].set_title(f'{method} Histogram', fontweight='bold')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)
    
    if n_trad < 6:
        axes[5].axis('off')
    
    fig3.suptitle('Histograms: Traditional Filters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("[5/6] Displaying DnCNN results...")
    n_dncnn = len(dncnn_results)
    fig4, axes = plt.subplots(1, n_dncnn, figsize=(6*n_dncnn, 5))
    if n_dncnn == 1:
        axes = [axes]
    
    for i, (method, img) in enumerate(dncnn_results.items()):
        axes[i].imshow(img, cmap='gray')
        
        psnr = calculate_psnr(ground_truth_gray if has_ground_truth else noisy_gray, img)
        if has_ground_truth:
            ssim = calculate_ssim(ground_truth_gray, img)
            title = f'{method}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}'
        else:
            title = f'{method}\nPSNR: {psnr:.2f} dB'
        
        axes[i].set_title(title, fontweight='bold', fontsize=14)
        axes[i].axis('off')
    
    fig4.suptitle('DnCNN Model Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("[6/6] Displaying histograms of DnCNN results...")
    fig5, axes = plt.subplots(1, n_dncnn, figsize=(6*n_dncnn, 4))
    if n_dncnn == 1:
        axes = [axes]
    
    for i, (method, img) in enumerate(dncnn_results.items()):
        axes[i].hist(img.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
        axes[i].set_title(f'{method} Histogram', fontweight='bold')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)
    
    fig5.suptitle('Histograms: DnCNN Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
    
    noisy_path = sys.argv[1]
    ground_truth_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Print header
    print("="*70)
    print("Single-Image All-Methods Denoising Tool")
    print("="*70)
    
    print(f"\nNoisy: {noisy_path}")
    if ground_truth_path:
        print(f"Ground Truth: {ground_truth_path}")
    
    noisy_gray, ground_truth_gray, filename_stem = load_images(noisy_path, ground_truth_path)
    
    print(f"Loaded grayscale image: {noisy_gray.shape}")
    if ground_truth_gray is not None:
        print(f"Loaded ground truth: {ground_truth_gray.shape}")
    else:
        print("No ground truth - visual comparison only")
    
    traditional_results, dncnn_results = apply_all_methods(noisy_gray, ground_truth_gray)
    visualize_all(noisy_gray, traditional_results, dncnn_results, ground_truth_gray, filename_stem)
    
    print("\n" + "="*70)
    print("Method Rankings (by PSNR):")
    print("="*70)
    
    all_results = {**traditional_results, **dncnn_results}
    metrics_data = []
    
    reference = ground_truth_gray if ground_truth_gray is not None else noisy_gray
    for method, img in all_results.items():
        psnr = calculate_psnr(reference, img)
        if ground_truth_gray is not None:
            ssim = calculate_ssim(ground_truth_gray, img)
            metrics_data.append((method, psnr, ssim))
        else:
            metrics_data.append((method, psnr, None))
    
    metrics_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, psnr, ssim) in enumerate(metrics_data, 1):
        best = " <-- BEST" if i == 1 else ""
        category = "[DnCNN]" if "DnCNN" in method else "[Trad] "
        if ssim is not None:
            print(f"{i:2d}. {category} {method:25s} - PSNR: {psnr:6.2f} dB, SSIM: {ssim:.3f}{best}")
        else:
            print(f"{i:2d}. {category} {method:25s} - PSNR: {psnr:6.2f} dB{best}")
    print("="*70)
    
    print("\nProcessing completed.")


if __name__ == '__main__':
    main()
