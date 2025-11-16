import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import sys



class ImageVisualizor:
    
    def __init__(self, original_image_path, noisey_image_path) -> None:
        self._original_images: List[np.ndarray] = []
        self._noisey_images: List[np.ndarray] = []
        
        self.original_image_path: Path = Path(original_image_path)
        self.noisey_image_path: Path = Path(noisey_image_path)
        
        
    def load_images(self) -> None:
        """Load matched pairs of images from both directories."""
        og_images: List[np.ndarray] = []
        noisey_images: List[np.ndarray] = []
        
        # Load multiple image formats
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        
        # Get original images with their filenames
        original_files = []
        for ext in image_extensions:
            original_files.extend(list(self.original_image_path.glob(ext)))
        
        original_files = sorted(original_files)
        
        print(f"Found {len(original_files)} original image(s)")
        
        # For each original image, find matching noisy image
        for orig_path in original_files:
            # Try to find matching noisy image
            # First try with same filename
            noisy_path = self.noisey_image_path / orig_path.name
            
            # If not found, try with "noisy_" prefix
            if not noisy_path.exists():
                noisy_path = self.noisey_image_path / f"noisy_{orig_path.name}"
            
            # If still not found, try other common prefixes
            if not noisy_path.exists():
                noisy_path = self.noisey_image_path / f"noise_{orig_path.name}"
            
            if noisy_path.exists():
                orig_img = cv2.imread(str(orig_path))
                noisy_img = cv2.imread(str(noisy_path))
                
                if orig_img is not None and noisy_img is not None:
                    # Check if dimensions match
                    if orig_img.shape == noisy_img.shape:
                        og_images.append(orig_img)
                        noisey_images.append(noisy_img)
                        print(f"✓ Loaded pair: {orig_path.name} (shape: {orig_img.shape})")
                    else:
                        print(f"✗ Skipped {orig_path.name}: dimension mismatch "
                              f"(orig: {orig_img.shape}, noisy: {noisy_img.shape})")
                else:
                    print(f"✗ Failed to load: {orig_path.name}")
            else:
                print(f"✗ No matching noisy image for: {orig_path.name}")
        
        self._original_images = og_images
        self._noisey_images = noisey_images
        
        print(f"\nSuccessfully loaded {len(self._original_images)} matching image pairs")
        
        if len(self._original_images) == 0:
            print("\nWarning: No matching image pairs found!")
            print("Make sure:")
            print("  1. Noisy images have the same filename as originals")
            print("  2. OR noisy images have 'noisy_' or 'noise_' prefix")
            print("  3. Both images have the same dimensions")
        
    @staticmethod
    def _get_image_stats(original_image: np.ndarray, noisey_image: np.ndarray) -> None:
        
        # Validate image dimensions match
        if original_image.shape != noisey_image.shape:
            raise ValueError(
                f"Image dimensions don't match! "
                f"Original: {original_image.shape}, Noisy: {noisey_image.shape}"
            )
        
        # Extract original image channels
        orig_b = original_image[:, :, 0].flatten()
        orig_g = original_image[:, :, 1].flatten()
        orig_r = original_image[:, :, 2].flatten()
        orig_total = original_image.flatten()
        
        # Calculate original image statistics
        orig_mean_b = np.mean(orig_b)
        orig_sigma_b = np.std(orig_b)
        
        orig_mean_g = np.mean(orig_g)
        orig_sigma_g = np.std(orig_g)
        
        orig_mean_r = np.mean(orig_r)
        orig_sigma_r = np.std(orig_r)
        
        orig_mean_total = np.mean(orig_total)
        orig_sigma_total = np.std(orig_total)
        
        # calculate the difference between og and noisey images
        noise = noisey_image.astype(np.float32) - original_image.astype(np.float32)
        
        # extract noise for each channel
        noise_b = noise[:, :, 0].flatten()
        noise_g = noise[:, :, 1].flatten()
        noise_r = noise[:, :, 2].flatten()
        noise_total = noise.flatten()
        
        # calculate noise statistics
        # BLUE channel noise statistics
        mean_b = np.mean(noise_b)
        sigma_b = np.std(noise_b)
        
        # GREEN channel statistics
        mean_g = np.mean(noise_g)
        sigma_g = np.std(noise_g)
        
        # RED channel statistics
        mean_r = np.mean(noise_r)
        sigma_r = np.std(noise_r)
        
        # TOTAL channels statistics
        mean_total = np.mean(noise_total)
        sigma_total = np.std(noise_total)
        
        
        print("========== Original Image Statistics ==========")
        print(f"\nBlue Channel Stats:")
        print(f"  Mean (μ): {orig_mean_b:.4f}")
        print(f"  Std Dev (σ): {orig_sigma_b:.4f}")
        print(f"  Min: {np.min(orig_b):.4f}")
        print(f"  Max: {np.max(orig_b):.4f}")

        print(f"\nGreen Channel Stats:")
        print(f"  Mean (μ): {orig_mean_g:.4f}")
        print(f"  Std Dev (σ): {orig_sigma_g:.4f}")
        print(f"  Min: {np.min(orig_g):.4f}")
        print(f"  Max: {np.max(orig_g):.4f}")

        print(f"\nRed Channel Stats:")
        print(f"  Mean (μ): {orig_mean_r:.4f}")
        print(f"  Std Dev (σ): {orig_sigma_r:.4f}")
        print(f"  Min: {np.min(orig_r):.4f}")
        print(f"  Max: {np.max(orig_r):.4f}")

        print(f"\nCombined (All Channels) Stats:")
        print(f"  Mean (μ): {orig_mean_total:.4f}")
        print(f"  Std Dev (σ): {orig_sigma_total:.4f}")
        print(f"  Min: {np.min(orig_total):.4f}")
        print(f"  Max: {np.max(orig_total):.4f}")
        print("=" * 48)
        
        print("\n========== Image Noise Statistics ==========")
        print(f"\nBlue Channel Stats:")
        print(f"  Mean (μ): {mean_b:.4f}")
        print(f"  Std Dev (σ): {sigma_b:.4f}")
        print(f"  Min: {np.min(noise_b):.4f}")
        print(f"  Max: {np.max(noise_b):.4f}")

        print(f"\nGreen Channel Stats:")
        print(f"  Mean (μ): {mean_g:.4f}")
        print(f"  Std Dev (σ): {sigma_g:.4f}")
        print(f"  Min: {np.min(noise_g):.4f}")
        print(f"  Max: {np.max(noise_g):.4f}")

        print(f"\nRed Channel Stats:")
        print(f"  Mean (μ): {mean_r:.4f}")
        print(f"  Std Dev (σ): {sigma_r:.4f}")
        print(f"  Min: {np.min(noise_r):.4f}")
        print(f"  Max: {np.max(noise_r):.4f}")

        print(f"\nCombined (All Channels) Stats:")
        print(f"  Mean (μ): {mean_total:.4f}")
        print(f"  Std Dev (σ): {sigma_total:.4f}")
        print(f"  Min: {np.min(noise_total):.4f}")
        print(f"  Max: {np.max(noise_total):.4f}")
        print("=" * 44)
        
    @staticmethod
    def _plot_image_histogram(original_image: np.ndarray, noisey_image: np.ndarray) -> Tuple[Figure, np.ndarray]:
        
        # Validate image dimensions match
        if original_image.shape != noisey_image.shape:
            raise ValueError(
                f"Image dimensions don't match! "
                f"Original: {original_image.shape}, Noisy: {noisey_image.shape}"
            )
        
        # Convert BGR to RGB for matplotlib display
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        noisey_rgb = cv2.cvtColor(noisey_image, cv2.COLOR_BGR2RGB)
        
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(21, 12))
        
        # Row 1: Image comparisons
        axs[0, 0].imshow(original_rgb)
        axs[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
        axs[0, 0].axis("off")
        
        axs[0, 1].imshow(noisey_rgb)
        axs[0, 1].set_title("Noisy Image", fontsize=12, fontweight='bold')
        axs[0, 1].axis("off")
        
        # Calculate the noise (difference between images)
        noise = noisey_image.astype(np.float32) - original_image.astype(np.float32)
        
        # Visualize noise (normalize for display)
        noise_vis = (noise - noise.min()) / (noise.max() - noise.min()) * 255
        noise_vis = noise_vis.astype(np.uint8)
        noise_vis_rgb = cv2.cvtColor(noise_vis, cv2.COLOR_BGR2RGB)
        
        axs[0, 2].imshow(noise_vis_rgb)
        axs[0, 2].set_title("Noise Visualization (Normalized)", fontsize=12, fontweight='bold')
        axs[0, 2].axis("off")
        
        # Extract noise channels for histogram plotting
        noise_b = noise[:, :, 0].flatten()
        noise_g = noise[:, :, 1].flatten()
        noise_r = noise[:, :, 2].flatten()
        noise_combined = noise.flatten()
        
        # row 2: Per-channel noise histograms with Gaussian overlays
        colors = ['blue', 'green', 'red']
        noise_channels = [noise_b, noise_g, noise_r]
        channel_names = ['Blue Channel', 'Green Channel', 'Red Channel']
        
        for i, (noise_ch, color, name) in enumerate(zip(noise_channels, colors, channel_names)):
            ax = axs[1, i]
            
            # lot histogram
            counts, bins, patches = ax.hist(noise_ch, bins=100, color=color, alpha=0.7,
                                           edgecolor='black', linewidth=0.5)
            
            # overlay Gaussian fit
            mu = np.mean(noise_ch)
            sigma = np.std(noise_ch)
            x = np.linspace(noise_ch.min(), noise_ch.max(), 100)
            gaussian_fit = len(noise_ch) * (bins[1] - bins[0]) * \
                          (1 / (sigma * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, gaussian_fit, 'k--', linewidth=2, label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')
            
            ax.set_title(f'{name} Noise Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel('Noise Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Row 3: Combined histogram, Q-Q plot, and statistics summary
        # Combined noise histogram
        ax = axs[2, 0]
        counts, bins, patches = ax.hist(noise_combined, bins=100, color='gray', alpha=0.7,
                                       edgecolor='black', linewidth=0.5)
        
        # Overlay Gaussian fit
        mu_all = np.mean(noise_combined)
        sigma_all = np.std(noise_combined)
        x = np.linspace(noise_combined.min(), noise_combined.max(), 100)
        gaussian_fit = len(noise_combined) * (bins[1] - bins[0]) * \
                      (1 / (sigma_all * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x - mu_all) / sigma_all) ** 2)
        ax.plot(x, gaussian_fit, 'r--', linewidth=2,
               label=f'Gaussian Fit\nμ={mu_all:.2f}, σ={sigma_all:.2f}')
        
        ax.set_title('Combined Noise Distribution (All Channels)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Noise Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Q-Q plot
        from scipy import stats
        ax = axs[2, 1]
        stats.probplot(noise_combined[::100], dist="norm", plot=ax)  # Sample every 100th point
        ax.set_title('Q-Q Plot (Normality Test)', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Statistics summary - spans last column
        ax = axs[2, 2]
        ax.axis('off')
        stats_text = f"""
                        NOISE STATISTICS
                        {'='*28}

                        Combined (All Channels):
                        Mean (μ):    {mu_all:>8.4f}
                        Std Dev (σ): {sigma_all:>8.4f}
                        Min:         {np.min(noise_combined):>8.4f}
                        Max:         {np.max(noise_combined):>8.4f}

                        Per Channel:
                        Blue:
                            μ = {np.mean(noise_b):>8.4f}
                            σ = {np.std(noise_b):>8.4f}
                        Green:
                            μ = {np.mean(noise_g):>8.4f}
                            σ = {np.std(noise_g):>8.4f}
                        Red:
                            μ = {np.mean(noise_r):>8.4f}
                            σ = {np.std(noise_r):>8.4f}

                        Expected:
                        μ ≈ 0, σ ≈ {sigma_all:.2f}
                    """
        ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
               verticalalignment='center', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Remove unused row 4
        for i in range(3):
            fig.delaxes(axs[3, i])
        
        plt.suptitle('Gaussian Noise Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, noise
    
    def analyze_images(self, num_samples: Optional[int] = None, interactive: bool = True) -> None:
        """
        Analyze and visualize image pairs.
        
        Parameters:
        -----------
        num_samples : int, optional
            Number of image pairs to analyze. If None, analyzes all images.
        interactive : bool, optional
            If True, prompts user between images. Default is True.
        """
        if not self._original_images or not self._noisey_images:
            print("Error: No images loaded. Call load_images() first.")
            return
        
        # Determine how many images to analyze
        total_images = min(len(self._original_images), len(self._noisey_images))
        if num_samples is None:
            num_samples = total_images
        else:
            num_samples = min(num_samples, total_images)
        
        print(f"\n{'='*60}")
        print(f"Analyzing {num_samples} image pair(s)...")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            original_img = self._original_images[i]
            noisy_img = self._noisey_images[i]
            
            print(f"\n--- Analyzing Image Pair {i+1}/{num_samples} ---")
            
            # Print statistics
            self._get_image_stats(original_img, noisy_img)
            
            # Generate and display plots
            fig, noise = self._plot_image_histogram(original_img, noisy_img)
            plt.show()
            
    
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}\n")
        
        
        
def main() -> None:
    """
    CLI tool for analyzing Gaussian noise in images.
    """
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Gaussian noise in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Analyze first 3 image pairs
            python visualization.py -o ./clean_images -n ./gaussian_noise_15_sigma -s 3
            
            # Analyze all image pairs
            python visualization.py -o ./output_images -n ./gaussian_noise_15_sigma
            
            # Analyze with different noise levels
            python visualization.py -o ./clean_images -n ./gaussian_noise_25_sigma -s 5
                """
    )
    
    parser.add_argument(
        '-o', '--original',
        type=str,
        required=True,
        help='Path to directory containing original images'
    )
    
    parser.add_argument(
        '-n', '--noisy',
        type=str,
        required=True,
        help='Path to directory containing noisy images'
    )
    
    parser.add_argument(
        '-s', '--samples',
        type=int,
        default=None,
        help='Number of image pairs to analyze (default: all images)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive prompts between images'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    original_path = Path(args.original)
    noisy_path = Path(args.noisy)
    
    if not original_path.exists():
        print(f"Error: Original image directory '{args.original}' does not exist.")
        sys.exit(1)
    
    if not noisy_path.exists():
        print(f"Error: Noisy image directory '{args.noisy}' does not exist.")
        sys.exit(1)
    
    # Create visualizer and run analysis
    print(f"\n{'='*60}")
    print("Gaussian Noise Analysis Tool")
    print(f"{'='*60}")
    print(f"Original images: {original_path.absolute()}")
    print(f"Noisy images: {noisy_path.absolute()}")
    print(f"{'='*60}\n")
    
    vis = ImageVisualizor(
        original_image_path=str(original_path),
        noisey_image_path=str(noisy_path)
    )
    
    # Load images
    vis.load_images()
    
    # Check if any images were loaded
    if not vis._original_images or not vis._noisey_images:
        print("Error: No images found in the specified directories.")
        sys.exit(1)
    
    # Analyze images
    vis.analyze_images(num_samples=args.samples, interactive=not args.no_interactive)
    
    
    
if __name__ == "__main__":
    main()
    

