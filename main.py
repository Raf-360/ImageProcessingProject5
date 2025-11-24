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
from utils.parameter_tuning import bayesian_optimize, get_param_space, adjust_params_for_constraints

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
    
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Automatically find optimal parameters using Bayesian optimization"
    )
    
    parser.add_argument(
        "--tune-metric",
        type=str,
        choices=['psnr', 'ssim'],
        default='psnr',
        help="Metric to optimize when auto-tuning (default: psnr)"
    )
    
    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=50,
        help="Number of optimization iterations for auto-tuning (default: 50)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to apply the filter iteratively (default: 1)"
    )
    
    return parser.parse_args()


def denoise_with_method(method_name: str, image, iterations: int = 1, **kwargs):
    """Apply specified denoising method iteratively.
    
    Args:
        method_name: Name of the denoising method
        image: Input image to denoise
        iterations: Number of times to apply the filter
        **kwargs: Additional parameters for the denoising method
    
    Returns:
        Tuple of (denoised_image, visualization_data)
    """
    result = image.copy()
    viz_data = None
    
    for i in range(iterations):
        if method_name == 'gaussian':
            result, _ = gaussian_denoise(result, **kwargs), None
        elif method_name == 'median':
            result, _ = median_denoise(result, **kwargs), None
        elif method_name == 'bilateral':
            result, _ = bilateral_denoise(result, **kwargs), None
        elif method_name == 'nlm':
            result, _ = nlm_denoise(result, **kwargs), None
        elif method_name == 'wiener':
            result, viz_data = wiener_denoise(result, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    return result, viz_data


def denoise_all_methods(image, iterations: int = 1):
    """Run all denoising methods with default parameters.
    
    Args:
        image: Input image to denoise
        iterations: Number of times to apply each filter
    
    Returns:
        Tuple of (results_dict, visualization_data)
    """
    results = {}
    viz_data = None
    
    # Apply each method iteratively
    for method_name in ['gaussian', 'median', 'bilateral', 'nlm', 'wiener']:
        result = image.copy()
        for i in range(iterations):
            if method_name == 'gaussian':
                result, _ = gaussian_denoise(result), None
            elif method_name == 'median':
                result, _ = median_denoise(result), None
            elif method_name == 'bilateral':
                result, _ = bilateral_denoise(result), None
            elif method_name == 'nlm':
                result, _ = nlm_denoise(result), None
            elif method_name == 'wiener':
                result, viz_data = wiener_denoise(result)
        
        results[method_name] = result
    
    return results, viz_data


def compare_methods(noisy_image, ground_truth, iterations: int = 1):
    """Compare all methods and return metrics DataFrame.
    
    Args:
        noisy_image: Noisy input image
        ground_truth: Clean reference image
        iterations: Number of times to apply each filter
    
    Returns:
        DataFrame with comparison results
    """
    methods = ['gaussian', 'median', 'bilateral', 'nlm', 'wiener']
    results_list = []
    
    for method in methods:
        start_time = time.time()
        
        denoised, _ = denoise_with_method(method, noisy_image, iterations=iterations)
        elapsed = time.time() - start_time
        
        # Get parameter descriptions
        params_desc = {
            'gaussian': 'kernel=(5,5), sigma=1.0',
            'median': 'kernel=5',
            'bilateral': 'd=15, sigma_c=35.71734057598922, sigma_s=200.0',
            'nlm': 'h=10, tw=7, sw=21',
            'wiener': 'mysize=7, noise_varience=0.01117'
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
        
        # Auto-tune parameters
        tuned_params = None
        if args.auto_tune:
            if args.method == 'all':
                print("\n⚠️  Warning: --auto-tune requires a specific method (-m). Using 'bilateral' as default.")
                args.method = 'bilateral'
            
            print("\n" + "=" * 60)
            print("🎯 Auto-Tuning Parameters")
            print("=" * 60)
            
            # Use first image for tuning
            noisy = noisy_images[0]
            clean = clean_images[0]
            
            # Get method function
            method_func_map = {
                'gaussian': gaussian_denoise,
                'median': median_denoise,
                'bilateral': bilateral_denoise,
                'nlm': nlm_denoise,
                'wiener': wiener_denoise
            }
            
            denoise_func = method_func_map[args.method]
            
            # Include iterations in tuning only if user didn't manually specify it
            include_iterations = (args.iterations == 1)  # Default value means auto-tune it
            param_space = get_param_space(args.method, include_iterations=include_iterations)
            
            if not param_space:
                print(f"❌ No parameter space defined for method '{args.method}'")
            else:
                # Pass current iterations value to bayesian_optimize
                tuned_params, best_psnr, best_ssim = bayesian_optimize(
                    denoise_func, noisy, clean, param_space,
                    args.method, n_calls=args.tune_iterations,
                    metric=args.tune_metric, iterations=args.iterations
                )
                
                # Extract tuned iterations if it was optimized
                tuned_iterations = tuned_params.pop('iterations', args.iterations)
                if include_iterations:
                    args.iterations = tuned_iterations
                
                # Adjust for constraints (odd kernel sizes, etc.)
                tuned_params = adjust_params_for_constraints(args.method, tuned_params)
                
                print("\n📊 Tuned Parameters Summary:")
                print(f"   Method: {args.method.upper()}")
                print(f"   Optimized for: {args.tune_metric.upper()}")
                print(f"   Best PSNR: {best_psnr:.2f} dB")
                print(f"   Best SSIM: {best_ssim:.4f}")
                print(f"   Parameters: {tuned_params}")
                if include_iterations:
                    print(f"   Optimal iterations: {args.iterations}")
                print(f"\n   These parameters will be used for all {num_to_process} images.\n")
        
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
            if args.iterations > 1:
                print(f"(Applying each filter {args.iterations} times)")
            print("=" * 60 + "\n")
            
            for i in range(num_to_process):
                noisy = noisy_images[i]
                clean = clean_images[i]
                
                print(f"Image {i+1}:")
                df = compare_methods(noisy, clean, iterations=args.iterations)
                print(df.to_string(index=False))
                print()
                
                # If visualize flag is set, show comparison for all images
                if args.visualize:
                    print(f"  📊 Generating visualization for image {i+1}...")
                    results, viz_data = denoise_all_methods(noisy, iterations=args.iterations)
                    visualize_results(noisy, results, clean, image_number=i+1, total_images=num_to_process)
                    
                    # Show Wiener filter frequency domain visualization
                    if viz_data is not None:
                        print(f"  📊 Showing Wiener filter frequency analysis for image {i+1}")
                        visualize_wiener_filter(**viz_data, image_number=i+1, total_images=num_to_process)
                    print()
        
        # Denoise and optionally save/visualize
        if args.method != 'all' or (args.visualize and not args.compare) or args.output:
            print("\n" + "=" * 60)
            print("🔧 Denoising Images")
            if args.iterations > 1:
                print(f"(Applying filter {args.iterations} times per image)")
            print("=" * 60 + "\n")
            
            for i in range(num_to_process):
                noisy = noisy_images[i]
                clean = clean_images[i]
                
                print(f"Processing image {i+1}/{num_to_process}...")
                
                if args.method == 'all':
                    results, viz_data = denoise_all_methods(noisy, iterations=args.iterations)
                else:
                    # Single method - use tuned params if available
                    if tuned_params:
                        denoised, viz_data = denoise_with_method(args.method, noisy, iterations=args.iterations, **tuned_params)
                    else:
                        denoised, viz_data = denoise_with_method(args.method, noisy, iterations=args.iterations)
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
                if args.visualize:
                    visualize_results(noisy, results, clean, image_number=i+1, total_images=num_to_process)
                    
                    # Show Wiener filter frequency domain visualization
                    if viz_data is not None:
                        print(f"  📊 Showing Wiener filter frequency analysis for image {i+1}")
                        visualize_wiener_filter(**viz_data, image_number=i+1, total_images=num_to_process)
                
                print()
        
        print("✅ Processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
