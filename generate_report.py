"""
Generate comprehensive reports with error maps and dataset-wide analysis.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from utils.image_io import load_image_pair
from utils.error_maps import visualize_error_map, visualize_multi_method_errors, visualize_noise_distribution
from utils.dataset_plots import (plot_metrics_distribution, plot_method_comparison,
                                 plot_psnr_ssim_scatter, plot_summary_statistics,
                                 plot_per_image_performance)
from utils.report_generation import generate_html_report, generate_pdf_report
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse
from utils.noise_estimation import estimate_noise_level
import time
import numpy as np
import cv2 as cv

from main import denoise_all_methods, denoise_with_method
from deep.inference import DnCNNDenoiser
from skimage.restoration import richardson_lucy
from skimage.filters import gaussian as sk_gaussian

def evaluate_all_methods(noisy_images: list, clean_images: list, 
                        dncnn_model_path: str = None, 
                        save_images: bool = False,
                        output_dir: Path = None,
                        noise_type: str = 'gaussian') -> pd.DataFrame:
    """
    Evaluate all denoising methods on provided images.
    
    Args:
        noisy_images: List of noisy images
        clean_images: List of clean images
        dncnn_model_path: Optional path to DnCNN model checkpoint
        save_images: Whether to save denoised images
        output_dir: Directory to save images to
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'motion_blur')
    
    Returns:
        DataFrame with evaluation results
    """
    # Select best traditional method based on noise type
    if noise_type == 'salt_pepper':
        methods = ['median']
    elif noise_type == 'gaussian':
        methods = ['bilateral']
    elif noise_type == 'motion_blur':
        methods = ['lucy_richardson']
    else:
        # Fallback to bilateral for unknown noise types
        methods = ['bilateral']
    
    results_list = []
    
    # Create images directory if saving
    images_dir = None
    if save_images and output_dir:
        images_dir = output_dir / "denoised_images"
        images_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for each method
        for method in methods:
            (images_dir / method).mkdir(exist_ok=True)
        
        # Create directories for original images
        (images_dir / "clean").mkdir(exist_ok=True)
        (images_dir / "noisy").mkdir(exist_ok=True)
    
    # Load DnCNN if model path provided
    dncnn_denoiser = None
    if dncnn_model_path:
        try:
            dncnn_denoiser = DnCNNDenoiser(dncnn_model_path)
            print(f"✓ Loaded DnCNN model from {dncnn_model_path}")
            if images_dir:
                (images_dir / "dncnn").mkdir(exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not load DnCNN model: {e}")
    
    for img_idx, (noisy, clean) in enumerate(zip(noisy_images, clean_images)):
        # Save clean and noisy images if requested
        if images_dir:
            cv.imwrite(str(images_dir / "clean" / f"image_{img_idx + 1}.png"), clean)
            cv.imwrite(str(images_dir / "noisy" / f"image_{img_idx + 1}.png"), noisy)
        
        for method in methods:
            start_time = time.time()
            
            # Denoise based on method
            if method == 'lucy_richardson':
                # Lucy-Richardson deconvolution for motion blur
                # Create motion blur PSF (Point Spread Function)
                psf = np.zeros((15, 15))
                psf[7, :] = 1.0  # Horizontal motion blur
                psf = psf / psf.sum()
                
                # Convert to float for processing
                noisy_float = noisy.astype(np.float64) / 255.0
                
                # Apply Lucy-Richardson per channel
                if len(noisy.shape) == 3:
                    denoised_float = np.zeros_like(noisy_float)
                    for c in range(3):
                        denoised_float[:, :, c] = richardson_lucy(noisy_float[:, :, c], psf, num_iter=30)
                else:
                    denoised_float = richardson_lucy(noisy_float, psf, num_iter=30)
                
                # Convert back to uint8
                denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)
            else:
                # Use existing methods
                denoised, _ = denoise_with_method(method, noisy)
            
            elapsed = time.time() - start_time
            
            # Save denoised image if requested
            if images_dir:
                cv.imwrite(str(images_dir / method / f"image_{img_idx + 1}.png"), denoised)
            
            # Calculate metrics
            psnr = calculate_psnr(denoised, clean)
            ssim = calculate_ssim(denoised, clean)
            mse = calculate_mse(denoised, clean)
            
            results_list.append({
                'Method': method,
                'Image': img_idx + 1,
                'PSNR': psnr,
                'SSIM': ssim,
                'MSE': mse,
                'Time': elapsed
            })
        
        # Evaluate DnCNN if available
        if dncnn_denoiser:
            start_time = time.time()
            
            # Convert image for model if needed
            model_channels = dncnn_denoiser.model.in_channels
            if model_channels == 1 and len(noisy.shape) == 3:
                noisy_for_model = cv.cvtColor(noisy, cv.COLOR_BGR2GRAY)
                clean_for_eval = cv.cvtColor(clean, cv.COLOR_BGR2GRAY)
            elif model_channels == 3 and len(noisy.shape) == 2:
                noisy_for_model = cv.cvtColor(noisy, cv.COLOR_GRAY2BGR)
                clean_for_eval = clean
            else:
                noisy_for_model = noisy
                clean_for_eval = clean
            
            # Denoise with DnCNN
            denoised = dncnn_denoiser.denoise(noisy_for_model)
            elapsed = time.time() - start_time
            
            # Save denoised image if requested
            if images_dir:
                # Convert back to BGR if needed for saving
                if len(denoised.shape) == 2:
                    denoised_to_save = cv.cvtColor(denoised, cv.COLOR_GRAY2BGR)
                else:
                    denoised_to_save = denoised
                cv.imwrite(str(images_dir / "dncnn" / f"image_{img_idx + 1}.png"), denoised_to_save)
            
            # Calculate metrics
            psnr = calculate_psnr(denoised, clean_for_eval)
            ssim = calculate_ssim(denoised, clean_for_eval)
            mse = calculate_mse(denoised, clean_for_eval)
            
            results_list.append({
                'Method': 'dncnn',
                'Image': img_idx + 1,
                'PSNR': psnr,
                'SSIM': ssim,
                'MSE': mse,
                'Time': elapsed
            })
    
    return pd.DataFrame(results_list)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive denoising analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-n", "--noisy",
        type=str,
        required=True,
        help="Path to folder containing noisy images"
    )
    
    parser.add_argument(
        "-c", "--clean",
        type=str,
        required=True,
        help="Path to folder containing clean/ground truth images"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./reports",
        help="Output directory for reports (default: ./reports)"
    )
    
    parser.add_argument(
        "--error-maps",
        action="store_true",
        help="Generate error map visualizations"
    )
    
    parser.add_argument(
        "--dataset-plots",
        action="store_true",
        help="Generate dataset-wide analysis plots"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--pdf-report",
        action="store_true",
        help="Generate PDF report"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all reports and visualizations"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to process (default: 5)"
    )
    
    parser.add_argument(
        "--dncnn-model",
        type=str,
        default=None,
        help="Path to DnCNN model checkpoint (optional)"
    )
    
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["gaussian", "salt_pepper", "motion_blur"],
        default="gaussian",
        help="Type of noise in images (default: gaussian)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set flags if --all is specified
    if args.all:
        args.error_maps = True
        args.dataset_plots = True
        args.html_report = True
        args.pdf_report = True
    
    noisy_path = Path(args.noisy)
    clean_path = Path(args.clean)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Comprehensive Analysis & Report Generation")
    print("=" * 70)
    
    # Validate paths
    if not noisy_path.exists():
        print(f"❌ Error: Directory '{noisy_path}' not found")
        sys.exit(1)
    
    if not clean_path.exists():
        print(f"❌ Error: Directory '{clean_path}' not found")
        sys.exit(1)
    
    try:
        # Load images
        print("\n📂 Loading images...")
        noisy_images, clean_images = load_image_pair(noisy_path, clean_path)
        num_to_process = min(args.num_images, len(noisy_images))
        print(f"✓ Loaded {num_to_process} image pairs\n")
        
        # Run evaluation to get results
        print("🔄 Evaluating all methods...")
        print(f"  Noise type: {args.noise_type}")
        results_df = evaluate_all_methods(
            noisy_images[:num_to_process], 
            clean_images[:num_to_process],
            dncnn_model_path=args.dncnn_model,
            save_images=True,
            output_dir=output_path,
            noise_type=args.noise_type
        )
        print(f"✓ Evaluation complete\n")
        
        # Generate image histograms with noise levels
        if args.error_maps or args.all:
            print("📊 Generating image histograms with noise analysis...")
            from utils.report_generation import generate_histogram_with_noise
            
            histograms_dir = output_path / "histograms"
            histograms_dir.mkdir(exist_ok=True)
            
            for idx in range(min(3, num_to_process)):
                hist_path = histograms_dir / f"histogram_image_{idx + 1}.png"
                generate_histogram_with_noise(
                    clean_images[idx], 
                    noisy_images[idx],
                    save_path=str(hist_path),
                    show=False
                )
            print(f"✓ Histograms saved to {histograms_dir}\n")
        
        # Generate error maps
        if args.error_maps:
            print("=" * 70)
            print("📊 Generating Error Maps")
            print("=" * 70)
            
            error_maps_dir = output_path / "error_maps"
            error_maps_dir.mkdir(exist_ok=True)
            
            for idx in range(min(3, num_to_process)):  # Generate for first 3 images
                print(f"\nProcessing image {idx + 1}...")
                noisy = noisy_images[idx]
                clean = clean_images[idx]
                
                # Get denoised results
                results, _ = denoise_all_methods(noisy)
                
                # Generate multi-method error map
                save_path = error_maps_dir / f"error_map_image_{idx + 1}.png"
                visualize_multi_method_errors(clean, noisy, results, save_path=str(save_path), show=False)
                
                # Generate noise distribution plot
                noise_save_path = error_maps_dir / f"noise_distribution_image_{idx + 1}.png"
                visualize_noise_distribution(noisy, clean, save_path=str(noise_save_path), show=False)
                
                # Generate noise distribution plot
                noise_save_path = error_maps_dir / f"noise_distribution_image_{idx + 1}.png"
                visualize_noise_distribution(noisy, clean, save_path=str(noise_save_path), show=False)
            
            print(f"\n✅ Error maps saved to {error_maps_dir}\n")
        
        # Generate dataset-wide plots
        if args.dataset_plots:
            print("=" * 70)
            print("📈 Generating Dataset-Wide Analysis Plots")
            print("=" * 70)
            
            plots_dir = output_path / "analysis_plots"
            plots_dir.mkdir(exist_ok=True)
            
            print("\n1. Generating metrics distribution plot...")
            plot_metrics_distribution(results_df, save_path=str(plots_dir / "metrics_distribution.png"))
            
            print("2. Generating method comparison plot...")
            plot_method_comparison(results_df, save_path=str(plots_dir / "method_comparison.png"))
            
            print("3. Generating PSNR vs SSIM scatter plot...")
            plot_psnr_ssim_scatter(results_df, save_path=str(plots_dir / "psnr_vs_ssim.png"))
            
            print("4. Generating summary statistics plot...")
            plot_summary_statistics(results_df, save_path=str(plots_dir / "summary_statistics.png"))
            
            if 'Image' in results_df.columns:
                print("5. Generating per-image performance plot...")
                plot_per_image_performance(results_df, metric='PSNR', 
                                          save_path=str(plots_dir / "per_image_psnr.png"))
                plot_per_image_performance(results_df, metric='SSIM',
                                          save_path=str(plots_dir / "per_image_ssim.png"))
            
            print(f"\n✅ Analysis plots saved to {plots_dir}\n")
        
        # Generate HTML report
        if args.html_report:
            print("=" * 70)
            print("📄 Generating HTML Report")
            print("=" * 70)
            
            html_path = output_path / "denoising_report.html"
            generate_html_report(results_df, str(html_path), 
                               include_plots=True,
                               project_name="Image Denoising Benchmark Report")
            print()
        
        # Generate PDF report
        if args.pdf_report:
            print("=" * 70)
            print("📑 Generating PDF Report")
            print("=" * 70)
            
            pdf_path = output_path / "denoising_report.pdf"
            generate_pdf_report(results_df, str(pdf_path),
                              project_name="Image Denoising Benchmark Report")
            print()
        
        print("=" * 70)
        print("✅ All reports generated successfully!")
        print(f"📁 Output directory: {output_path.absolute()}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
