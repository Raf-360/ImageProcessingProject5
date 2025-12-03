"""
DnCNN Image Denoising Demo Tool

Simple demonstration tool that compares traditional filtering methods with DnCNN.
Processes 5 image pairs and displays side-by-side comparisons with metrics.

Usage:
    python demo.py                                              # Use default paths
    python demo.py path/to/noisy/ path/to/clean/              # Custom paths
    python demo.py --report                                     # Generate HTML report
    python demo.py path/to/noisy/ path/to/clean/ --report     # Custom paths + report
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.metrics import calculate_psnr, calculate_ssim
from utils.image_io import load_images
from utils.report_generation import generate_html_report
from deep.inference import DnCNNDenoiser
import pandas as pd
from skimage.restoration import richardson_lucy

# Traditional filters
from traditional.gaussian import gaussian_denoise
from traditional.median import median_denoise
from traditional.bilateral import bilateral_denoise
from traditional.nlm import nlm_denoise
from traditional.wiener import wiener_denoise


# Default paths
DEFAULT_CLEAN_FOLDER = "data/test/xray/clean"
DEFAULT_NOISY_FOLDER = "data/test/xray/gaussian_noise_15_sigma"

# Noise type mapping
NOISE_TYPES = {
    '1': ('gaussian', 'Gaussian'),
    '2': ('salt_pepper', 'Salt & Pepper'),
    '3': ('motion_blur', 'Motion Blur')
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DnCNN Image Denoising Demo - Compare traditional filters with DnCNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                                    # Use default test images
  python demo.py noisy/ clean/                     # Custom image folders
  python demo.py noisy/ clean/ --report            # Generate HTML report

Default paths:
  Clean:  data/test/xray/clean
  Noisy:  data/test/xray/gaussian_noise_15_sigma
        """
    )
    
    parser.add_argument(
        'noisy_folder',
        nargs='?',
        default=DEFAULT_NOISY_FOLDER,
        help=f'Path to noisy images folder (default: {DEFAULT_NOISY_FOLDER})'
    )
    
    parser.add_argument(
        'clean_folder',
        nargs='?',
        default=DEFAULT_CLEAN_FOLDER,
        help=f'Path to clean images folder (default: {DEFAULT_CLEAN_FOLDER})'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML report (convertible to PDF)'
    )
    
    return parser.parse_args()


def match_image_pairs(noisy_folder: Path, clean_folder: Path, limit: int = 5) -> List[Tuple[Path, Path]]:
    """
    Match noisy and clean image pairs by filename.
    
    Args:
        noisy_folder: Path to noisy images
        clean_folder: Path to clean images
        limit: Maximum number of pairs to return
    
    Returns:
        List of (noisy_path, clean_path) tuples
    """
    # Get image files
    extensions = {'.png', '.jpg', '.jpeg'}
    
    clean_files = {f.name: f for f in clean_folder.iterdir() 
                   if f.suffix.lower() in extensions}
    
    noisy_files = {}
    for f in noisy_folder.iterdir():
        if f.suffix.lower() in extensions:
            # Handle 'noisy_' prefix
            clean_name = f.name.replace('noisy_', '', 1)
            noisy_files[clean_name] = f
    
    # Match pairs
    common_names = set(clean_files.keys()) & set(noisy_files.keys())
    
    if not common_names:
        print("ERROR: No matching image pairs found!")
        print(f"  Clean folder: {clean_folder}")
        print(f"  Noisy folder: {noisy_folder}")
        if clean_files:
            print(f"  Example clean: {list(clean_files.keys())[0]}")
        if noisy_files:
            print(f"  Example noisy: {list(noisy_files.keys())[0]}")
        sys.exit(1)
    
    pairs = [(noisy_files[name], clean_files[name]) for name in sorted(common_names)]
    
    return pairs[:limit]


def load_traditional_filters() -> Dict[str, callable]:
    """
    Load available traditional filtering methods.
    Silently skips any that fail to import.
    
    Returns:
        Dictionary of {name: filter_function}
    """
    filters = {}
    
    filter_configs = [
        ('Gaussian', gaussian_denoise),
        ('Median', median_denoise),
        ('Bilateral', bilateral_denoise),
        ('NLM', nlm_denoise),
        ('Wiener', wiener_denoise)
    ]
    
    for name, func in filter_configs:
        try:
            # Test if function is callable
            if callable(func):
                filters[name] = func
        except Exception:
            pass  # Silently skip
    
    return filters


def run_traditional_filters(noisy_img: np.ndarray, clean_img: np.ndarray, 
                           noise_type: str) -> List[Tuple[str, np.ndarray, float, float]]:
    """
    Run the best traditional filter for the specified noise type.
    
    Args:
        noisy_img: Noisy image
        clean_img: Ground truth clean image
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'motion_blur')
    
    Returns:
        List of (name, denoised_img, psnr, ssim) with single best method
    """
    results = []
    
    try:
        # Select best method based on noise type
        if noise_type == 'salt_pepper':
            denoised = median_denoise(noisy_img)
            method_name = 'Median'
        elif noise_type == 'gaussian':
            denoised = bilateral_denoise(noisy_img)
            method_name = 'Bilateral'
        elif noise_type == 'motion_blur':
            # Lucy-Richardson deconvolution
            psf = np.zeros((15, 15))
            psf[7, :] = 1.0  # Horizontal motion blur
            psf = psf / psf.sum()
            
            # Convert to float for processing
            noisy_float = noisy_img.astype(np.float64) / 255.0
            
            # Apply Lucy-Richardson per channel
            if len(noisy_img.shape) == 3:
                denoised_float = np.zeros_like(noisy_float)
                for c in range(3):
                    denoised_float[:, :, c] = richardson_lucy(noisy_float[:, :, c], psf, num_iter=30)
            else:
                denoised_float = richardson_lucy(noisy_float, psf, num_iter=30)
            
            # Convert back to uint8
            denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
            method_name = 'Lucy-Richardson'
        else:
            # Fallback to bilateral
            denoised = bilateral_denoise(noisy_img)
            method_name = 'Bilateral'
        
        # Compute metrics
        psnr = calculate_psnr(denoised, clean_img)
        ssim = calculate_ssim(denoised, clean_img)
        
        results.append((method_name, denoised, psnr, ssim))
    except Exception as e:
        print(f"  ⚠️  Filter failed: {e}")
    
    return results


def select_noise_type() -> Tuple[str, str]:
    """
    Interactive noise type selection.
    
    Returns:
        Tuple of (model_filename_prefix, display_name)
    """
    print("\nSelect noise type:")
    print("  [1] Gaussian")
    print("  [2] Salt & Pepper")
    print("  [3] Motion Blur")
    
    while True:
        choice = input("\nChoice (1-3): ").strip()
        if choice in NOISE_TYPES:
            return NOISE_TYPES[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")


def load_dncnn_model(noise_type: str) -> DnCNNDenoiser:
    """
    Load DnCNN model for specified noise type.
    
    Args:
        noise_type: Noise type identifier ('gaussian', 'salt_pepper', 'motion_blur')
    
    Returns:
        DnCNNDenoiser instance
    """
    # Try specific model first
    model_path = Path(f"models/{noise_type}_dncnn.pth")
    
    if not model_path.exists():
        # Fallback to general checkpoint
        model_path = Path("checkpoints/checkpoint_best.pth")
        if not model_path.exists():
            print(f"\nERROR: No DnCNN model found!")
            print(f"  Tried: models/{noise_type}_dncnn.pth")
            print(f"  Tried: checkpoints/checkpoint_best.pth")
            print(f"\nPlease train a model first or place a checkpoint in models/")
            sys.exit(1)
        print(f"Note: Using fallback model from checkpoints/")
    
    # Load model
    denoiser = DnCNNDenoiser(str(model_path))
    
    # Log model type
    channels = denoiser.model.in_channels
    model_type = "RGB (3-channel)" if channels == 3 else "Grayscale (1-channel)"
    print(f"Loaded {model_type} DnCNN model")
    
    return denoiser


def convert_image_for_model(image: np.ndarray, model_channels: int) -> np.ndarray:
    """
    Convert image to match model's expected channels.
    
    Args:
        image: Input image (H,W) or (H,W,C)
        model_channels: Expected channels (1 or 3)
    
    Returns:
        Converted image
    """
    img_channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    if img_channels == model_channels:
        return image
    
    if model_channels == 1 and img_channels == 3:
        # RGB to grayscale
        print("  Converting RGB → grayscale for 1-channel model")
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif model_channels == 3 and img_channels == 1:
        # Grayscale to RGB
        print("  Converting grayscale → RGB for 3-channel model")
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    return image


def visualize_comparison(clean_img: np.ndarray, noisy_img: np.ndarray,
                        traditional_name: str, traditional_img: np.ndarray, traditional_psnr: float, traditional_ssim: float,
                        dncnn_img: np.ndarray, dncnn_psnr: float, dncnn_ssim: float,
                        image_name: str):
    """
    Display 4-column comparison with metrics.
    
    Args:
        All images should be in the same format (RGB or grayscale)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Denoising Comparison - {image_name}', fontsize=14, fontweight='bold')
    
    # Determine if grayscale
    is_gray = len(clean_img.shape) == 2
    cmap = 'gray' if is_gray else None
    
    # Compute noisy metrics
    noisy_psnr = calculate_psnr(noisy_img, clean_img)
    noisy_ssim = calculate_ssim(noisy_img, clean_img)
    
    # Images and titles
    images = [clean_img, noisy_img, traditional_img, dncnn_img]
    titles = ['Clean (Ground Truth)', f'Noisy', f'Best Traditional\n({traditional_name})', 'DnCNN']
    metrics = [
        ('', ''),
        (f'PSNR: {noisy_psnr:.2f} dB', f'SSIM: {noisy_ssim:.4f}'),
        (f'PSNR: {traditional_psnr:.2f} dB', f'SSIM: {traditional_ssim:.4f}'),
        (f'PSNR: {dncnn_psnr:.2f} dB', f'SSIM: {dncnn_ssim:.4f}')
    ]
    
    # Find best method
    best_idx = 2 if traditional_psnr > dncnn_psnr else 3
    
    for idx, (ax, img, title, (psnr_text, ssim_text)) in enumerate(zip(axes, images, titles, metrics)):
        # Display image
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        else:
            # Convert BGR to RGB for display
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Add metrics text
        if psnr_text:
            ax.text(0.5, -0.1, psnr_text, transform=ax.transAxes,
                   ha='center', fontsize=9)
            ax.text(0.5, -0.15, ssim_text, transform=ax.transAxes,
                   ha='center', fontsize=9)
        
        # Highlight best method with green border
        if idx == best_idx:
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                     fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demo execution."""
    args = parse_args()
    
    # Validate paths
    noisy_folder = Path(args.noisy_folder)
    clean_folder = Path(args.clean_folder)
    
    if not noisy_folder.exists():
        print(f"ERROR: Noisy folder not found: {noisy_folder}")
        sys.exit(1)
    
    if not clean_folder.exists():
        print(f"ERROR: Clean folder not found: {clean_folder}")
        sys.exit(1)
    
    print("=" * 70)
    print("DnCNN Image Denoising Demo")
    print("=" * 70)
    
    # Match image pairs
    print(f"\nLoading images...")
    print(f"  Clean: {clean_folder}")
    print(f"  Noisy: {noisy_folder}")
    
    pairs = match_image_pairs(noisy_folder, clean_folder, limit=5)
    print(f"  Found {len(pairs)} image pairs (processing 5)")
    
    # Load traditional filters
    filters = load_traditional_filters()
    print(f"\nLoaded {len(filters)}/5 traditional filters")
    
    # Select noise type and load DnCNN
    noise_type, noise_display = select_noise_type()
    print(f"\nSelected noise type: {noise_display}")
    
    denoiser = load_dncnn_model(noise_type)
    model_channels = denoiser.model.in_channels
    
    # Process images
    print(f"\n{'='*70}")
    print("Processing Images")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for idx, (noisy_path, clean_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {clean_path.name}")
        
        # Load images
        clean_img = cv.imread(str(clean_path), cv.IMREAD_COLOR)
        noisy_img = cv.imread(str(noisy_path), cv.IMREAD_COLOR)
        
        if clean_img is None or noisy_img is None:
            print(f"  ERROR: Failed to load images, skipping...")
            continue
        
        # Convert for model if needed
        clean_for_model = convert_image_for_model(clean_img, model_channels)
        noisy_for_model = convert_image_for_model(noisy_img, model_channels)
        
        # Run best traditional filter for noise type
        traditional_results = run_traditional_filters(noisy_img, clean_img, noise_type)
        
        if not traditional_results:
            print(f"  ERROR: Traditional filter failed, skipping...")
            continue
        
        best_trad_name, best_trad_img, best_trad_psnr, best_trad_ssim = traditional_results[0]
        
        # Run DnCNN
        dncnn_img = denoiser.denoise(noisy_for_model)
        
        # Convert back if needed
        if model_channels != 3:
            dncnn_img = cv.cvtColor(dncnn_img, cv.COLOR_GRAY2BGR)
        
        dncnn_psnr = calculate_psnr(dncnn_img, clean_img)
        dncnn_ssim = calculate_ssim(dncnn_img, clean_img)
        
        # Store results
        all_results.append({
            'name': clean_path.name,
            'traditional_name': best_trad_name,
            'traditional_psnr': best_trad_psnr,
            'traditional_ssim': best_trad_ssim,
            'dncnn_psnr': dncnn_psnr,
            'dncnn_ssim': dncnn_ssim
        })
        
        # Visualize
        visualize_comparison(clean_img, noisy_img,
                           best_trad_name, best_trad_img, best_trad_psnr, best_trad_ssim,
                           dncnn_img, dncnn_psnr, dncnn_ssim,
                           clean_path.name)
    
    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}\n")
        
        avg_trad_psnr = np.mean([r['traditional_psnr'] for r in all_results])
        avg_trad_ssim = np.mean([r['traditional_ssim'] for r in all_results])
        avg_dncnn_psnr = np.mean([r['dncnn_psnr'] for r in all_results])
        avg_dncnn_ssim = np.mean([r['dncnn_ssim'] for r in all_results])
        
        print(f"Best Traditional (avg): PSNR: {avg_trad_psnr:.2f} dB, SSIM: {avg_trad_ssim:.4f}")
        print(f"DnCNN (avg):            PSNR: {avg_dncnn_psnr:.2f} dB, SSIM: {avg_dncnn_ssim:.4f}")
        print(f"\nDnCNN Improvement:      +{avg_dncnn_psnr - avg_trad_psnr:.2f} dB PSNR, "
              f"+{avg_dncnn_ssim - avg_trad_ssim:.4f} SSIM")
        
        if args.report:
            print(f"\n{'='*70}")
            print("Generating HTML Report")
            print(f"{'='*70}\n")
            
            # Create results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = Path(f"results/demo_{timestamp}")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare DataFrame for report generation
            df_data = []
            for r in all_results:
                # Add traditional method row
                df_data.append({
                    'Image': r['name'],
                    'Method': r['traditional_name'].lower(),
                    'PSNR': r['traditional_psnr'],
                    'SSIM': r['traditional_ssim']
                })
                # Add DnCNN row
                df_data.append({
                    'Image': r['name'],
                    'Method': 'dncnn',
                    'PSNR': r['dncnn_psnr'],
                    'SSIM': r['dncnn_ssim']
                })
            
            results_df = pd.DataFrame(df_data)
            
            # Generate HTML report
            report_path = result_dir / "report.html"
            generate_html_report(
                results_df=results_df,
                output_path=str(report_path),
                include_plots=True,
                project_name=f"DnCNN Demo - {noise_display} Noise"
            )
            
            print(f"\nTo convert to PDF:")
            print(f"  1. Open {report_path} in your browser")
            print(f"  2. Print → Save as PDF")
    
    print(f"\n{'='*70}")
    print("Demo Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
