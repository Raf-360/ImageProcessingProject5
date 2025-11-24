"""
Parameter tuning utilities using Bayesian optimization.
"""

import numpy as np
from typing import Dict, Tuple, Callable
from skopt import gp_minimize
from skopt.space import Integer, Real

from utils.metrics import calculate_psnr, calculate_ssim


def bayesian_optimize(denoise_func: Callable, noisy: np.ndarray, clean: np.ndarray,
                     param_space: Dict, method_name: str, n_calls: int = 50,
                     metric: str = 'psnr', iterations: int = 1) -> Tuple[Dict, float, float]:
    """
    Use Bayesian optimization to find optimal denoising parameters.
    
    Args:
        denoise_func: Denoising function to optimize
        noisy: Noisy image
        clean: Ground truth image
        param_space: Dict of parameter names to (min, max, type) tuples
        method_name: Name of the method being optimized
        n_calls: Number of optimization iterations
        metric: 'psnr' or 'ssim' to optimize
        iterations: Number of times to apply the filter iteratively
    
    Returns:
        best_params, best_psnr, best_ssim
    """
    print(f"\n🔍 Auto-tuning {method_name.upper()} parameters using Bayesian Optimization...")
    print(f"   Metric to optimize: {metric.upper()}")
    print(f"   Number of optimization calls: {n_calls}")
    if iterations > 1:
        print(f"   Filter iterations per test: {iterations}")
    print(f"   This may take a few minutes...\n")
    
    # Build skopt search space
    param_names = list(param_space.keys())
    space = []
    
    for name, spec in param_space.items():
        min_val, max_val, param_type = spec
        if param_type == 'int':
            space.append(Integer(min_val, max_val, name=name))
        else:  # float
            space.append(Real(min_val, max_val, name=name))
    
    # Track best result
    best_result = {'psnr': 0, 'ssim': 0, 'params': {}}
    iteration = [0]
    
    def objective(params):
        """Objective function to minimize (negative score)."""
        iteration[0] += 1
        param_dict = dict(zip(param_names, params))
        
        # Convert to appropriate types
        for name, spec in param_space.items():
            if spec[2] == 'int':
                param_dict[name] = int(param_dict[name])
        
        try:
            # Apply denoising function iteratively
            result_img = noisy.copy()
            for _ in range(iterations):
                result = denoise_func(result_img, **param_dict)
                
                # Handle methods that return (image, viz_data) tuple
                if isinstance(result, tuple):
                    result_img = result[0]
                else:
                    result_img = result
            
            psnr = calculate_psnr(result_img, clean)
            ssim = calculate_ssim(result_img, clean)
            
            score = psnr if metric == 'psnr' else ssim
            
            # Track best
            if score > best_result[metric]:
                best_result['psnr'] = psnr
                best_result['ssim'] = ssim
                best_result['params'] = param_dict.copy()
                print(f"   Iter {iteration[0]}/{n_calls}: ✓ New best! {param_dict}")
                print(f"              PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            elif iteration[0] % 10 == 0:
                print(f"   Iter {iteration[0]}/{n_calls}: {param_dict} -> PSNR={psnr:.2f}, SSIM={ssim:.4f}")
            
            return -score  # Minimize negative = maximize positive
            
        except Exception as e:
            print(f"   Iter {iteration[0]}/{n_calls}: Failed with {param_dict} - {e}")
            return 1e6  # Return large penalty for failed configurations
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=False,
        n_initial_points=10  # Random exploration first
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"   Best PSNR: {best_result['psnr']:.2f} dB")
    print(f"   Best SSIM: {best_result['ssim']:.4f}")
    print(f"   Best parameters: {best_result['params']}\n")
    
    return best_result['params'], best_result['psnr'], best_result['ssim']


def get_param_space(method_name: str, include_iterations: bool = True) -> Dict:
    """
    Get parameter search space for each method.
    
    Args:
        method_name: Name of the denoising method
        include_iterations: Whether to include iterations in the search space
    
    Returns:
        Dict mapping parameter names to (min, max, type) tuples
    """
    param_spaces = {
        'gaussian': {
            'kernel_size': (3, 15, 'int'),  # Must be odd, will adjust
            'sigma': (0.5, 3.0, 'float')
        },
        'median': {
            'kernel_size': (3, 15, 'int')  # Must be odd
        },
        'bilateral': {
            'd': (5, 15, 'int'),
            'sigma_color': (25, 200, 'float'),
            'sigma_space': (25, 200, 'float')
        },
        'nlm': {
            'h': (3, 20, 'float'),
            'template_window_size': (5, 11, 'int'),  # Must be odd
            'search_window_size': (15, 35, 'int')  # Must be odd
        },
        'wiener': {
            'mysize': (3, 11, 'int'),  # Must be odd
            'noise_variance': (0.001, 0.1, 'float')
        }
    }
    
    space = param_spaces.get(method_name, {})
    
    # Optionally add iterations to the search space
    if include_iterations and space:
        space['iterations'] = (1, 5, 'int')
    
    return space


def adjust_params_for_constraints(method_name: str, params: Dict) -> Dict:
    """
    Adjust parameters to meet method-specific constraints (e.g., odd kernel sizes).
    
    Args:
        method_name: Name of the denoising method
        params: Dictionary of parameters
    
    Returns:
        Adjusted parameters dictionary
    """
    adjusted = params.copy()
    
    # Ensure odd kernel sizes
    if method_name == 'gaussian' and 'kernel_size' in adjusted:
        ks = adjusted['kernel_size']
        adjusted['kernel_size'] = ks if ks % 2 == 1 else ks + 1
    
    if method_name == 'median' and 'kernel_size' in adjusted:
        ks = adjusted['kernel_size']
        adjusted['kernel_size'] = ks if ks % 2 == 1 else ks + 1
    
    if method_name == 'nlm':
        if 'template_window_size' in adjusted:
            tw = adjusted['template_window_size']
            adjusted['template_window_size'] = tw if tw % 2 == 1 else tw + 1
        if 'search_window_size' in adjusted:
            sw = adjusted['search_window_size']
            adjusted['search_window_size'] = sw if sw % 2 == 1 else sw + 1
    
    if method_name == 'wiener' and 'mysize' in adjusted:
        ms = adjusted['mysize']
        adjusted['mysize'] = ms if ms % 2 == 1 else ms + 1
    
    return adjusted
