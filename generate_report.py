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
import time
import numpy as np

from main import denoise_all_methods, denoise_with_method

def evaluate_all_methods(noisy_images: list, clean_images: list) -> pd.DataFrame:
    """
    Evaluate all denoising methods on provided images.
    
    Args:
        noisy_images: List of noisy images
        clean_images: List of clean images
    
    Returns:
        DataFrame with evaluation results
    """
    methods = ['gaussian', 'median', 'bilateral', 'nlm', 'wiener']
    results_list = []
    
    for img_idx, (noisy, clean) in enumerate(zip(noisy_images, clean_images)):
        for method in methods:
            start_time = time.time()
            
            # Denoise
            denoised, _ = denoise_with_method(method, noisy)
            elapsed = time.time() - start_time
            
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
        results_df = evaluate_all_methods(noisy_images[:num_to_process], 
                                         clean_images[:num_to_process])
        print(f"✓ Evaluation complete\n")
        
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
