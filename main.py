"""
Main entry point for traditional image denoising.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import time

from utils.image_io import load_image_pair, save_image
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import visualize_results, visualize_wiener_filter
from utils.noise_estimation import estimate_noise_level

from traditional.gaussian import gaussian_denoise
from traditional.median import median_denoise
from traditional.bilateral import bilateral_denoise
from traditional.nlm import nlm_denoise
from traditional.wiener import wiener_denoise


AVAILABLE_METHODS = ['gaussian', 'median', 'bilateral', 'nlm', 'wiener', 'all']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traditional Image Denoising Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Compare all methods on first image
  %(prog)s -n data/noisy -c data/clean --compare
  
  # Denoise with specific method and visualize
  %(prog)s -n data/noisy -c data/clean -m bilateral --visualize
  
  # Process multiple images and save results
  %(prog)s -n data/noisy -c data/clean -m nlm -o output_folder --num-images 10
        """
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
        "-m", "--method",
        type=str,
        choices=AVAILABLE_METHODS,
        default='all',
        help="Denoising method to use (default: all)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output folder to save denoised images"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to process (default: 5)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all denoising methods and show metrics"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize denoising results"
    )
    
    parser.add_argument(
        "--estimate-noise",
        action="store_true",
        help="Estimate noise level in images"
    )
    
    return parser.parse_args()


def denoise_with_method(method_name: str, image, **kwargs):
    """Apply specified denoising method."""
    if method_name == 'gaussian':
        return gaussian_denoise(image, **kwargs), None
    elif method_name == 'median':
        return median_denoise(image, **kwargs), None
    elif method_name == 'bilateral':
        return bilateral_denoise(image, **kwargs), None
    elif method_name == 'nlm':
        return nlm_denoise(image, **kwargs), None
    elif method_name == 'wiener':
        return wiener_denoise(image, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def denoise_all_methods(image):
    """Run all denoising methods with default parameters."""
    results = {}
    viz_data = None
    
    results['gaussian'], _ = gaussian_denoise(image), None
    results['median'], _ = median_denoise(image), None
    results['bilateral'], _ = bilateral_denoise(image), None
    results['nlm'], _ = nlm_denoise(image), None
    results['wiener'], viz_data = wiener_denoise(image)
    
    return results, viz_data


def compare_methods(noisy_image, ground_truth):
    """Compare all methods and return metrics DataFrame."""
    methods = ['gaussian', 'median', 'bilateral', 'nlm', 'wiener']
    results_list = []
    
    for method in methods:
        start_time = time.time()
        
        denoised, _ = denoise_with_method(method, noisy_image)
        elapsed = time.time() - start_time
        
        # Get parameter descriptions
        params_desc = {
            'gaussian': 'kernel=(5,5), sigma=1.0',
            'median': 'kernel=5',
            'bilateral': 'd=9, sigma_c=75, sigma_s=75',
            'nlm': 'h=10, tw=7, sw=21',
            'wiener': 'mysize=5, auto noise'
        }
        
        result = {
            'Method': method.upper(),
            'Time (s)': f'{elapsed:.3f}',
            'Parameters': params_desc[method]
        }
        
        if ground_truth is not None:
            result['PSNR (dB)'] = f"{calculate_psnr(denoised, ground_truth):.2f}"
            result['SSIM'] = f"{calculate_ssim(denoised, ground_truth):.4f}"
        
        results_list.append(result)
    
    return pd.DataFrame(results_list)


def main():
    args = parse_args()
    
    noisy_path = Path(args.noisy)
    clean_path = Path(args.clean)
    
    # Validate paths
    if not noisy_path.exists():
        print(f"❌ Error: Directory '{noisy_path}' not found")
        sys.exit(1)
    
    if not clean_path.exists():
        print(f"❌ Error: Directory '{clean_path}' not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Traditional Image Denoising Toolkit")
    print("=" * 60)
    
    try:
        # Load images
        print("\n📂 Loading images...")
        noisy_images, clean_images = load_image_pair(noisy_path, clean_path)
        
        print(f"✓ Loaded {len(noisy_images)} noisy images")
        print(f"✓ Loaded {len(clean_images)} ground truth images\n")
        
        if len(noisy_images) == 0:
            print("❌ Error: No noisy images loaded!")
            sys.exit(1)
        
        num_to_process = min(args.num_images, len(noisy_images))
        
        # Estimate noise level
        if args.estimate_noise:
            print("\n" + "=" * 60)
            print("📊 Estimated Noise Levels")
            print("=" * 60)
            for i in range(num_to_process):
                print(f"\nImage {i+1}:")
                estimate_noise_level(noisy_images[i])
        
        # Compare methods
        if args.compare:
            print("\n" + "=" * 60)
            print("📊 Method Comparison")
            print("=" * 60 + "\n")
            
            for i in range(num_to_process):
                noisy = noisy_images[i]
                clean = clean_images[i]
                
                print(f"Image {i+1}:")
                df = compare_methods(noisy, clean)
                print(df.to_string(index=False))
                print()
                
                # If visualize flag is set, show comparison for first image
                if args.visualize and i == 0:
                    print(f"  📊 Generating visualization for image {i+1}...")
                    results, viz_data = denoise_all_methods(noisy)
                    visualize_results(noisy, results, clean)
                    
                    # Show Wiener filter frequency domain visualization
                    if viz_data is not None:
                        print(f"  📊 Showing Wiener filter frequency analysis")
                        visualize_wiener_filter(**viz_data)
                    print()
        
        # Denoise and optionally save/visualize
        if args.method != 'all' or (args.visualize and not args.compare) or args.output:
            print("\n" + "=" * 60)
            print("🔧 Denoising Images")
            print("=" * 60 + "\n")
            
            for i in range(num_to_process):
                noisy = noisy_images[i]
                clean = clean_images[i]
                
                print(f"Processing image {i+1}/{num_to_process}...")
                
                if args.method == 'all':
                    results, viz_data = denoise_all_methods(noisy)
                else:
                    # Single method
                    denoised, viz_data = denoise_with_method(args.method, noisy)
                    results = {args.method: denoised}
                
                # Calculate metrics
                for method_name, denoised_img in results.items():
                    psnr = calculate_psnr(denoised_img, clean)
                    ssim = calculate_ssim(denoised_img, clean)
                    print(f"  {method_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
                
                # Save results
                if args.output:
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    for method_name, denoised_img in results.items():
                        save_path = output_path / f"{method_name}_image_{i+1}.png"
                        save_image(denoised_img, save_path)
                
                # Visualize
                if args.visualize and i == 0:
                    visualize_results(noisy, results, clean)
                    
                    # Show Wiener filter frequency domain visualization
                    if viz_data is not None:
                        print(f"  📊 Showing Wiener filter frequency analysis for image {i+1}")
                        visualize_wiener_filter(**viz_data)
                
                print()
        
        print("✅ Processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
