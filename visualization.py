import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_image_pair(original_path, noisy_path):
    """Load original and noisy image pair."""
    original = cv2.imread(str(original_path))
    noisy = cv2.imread(str(noisy_path))
    return original, noisy

def plot_gaussian_noise_distribution(original_img, noisy_img, title="Gaussian Noise Distribution"):
    """
    Visualize the distribution of Gaussian noise added to an image.
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        Original clean image (BGR format)
    noisy_img : numpy.ndarray
        Noisy image (BGR format)
    title : str
        Title for the plot
    """
    # Calculate the noise (difference between noisy and original)
    noise = noisy_img.astype(np.float32) - original_img.astype(np.float32)
    
    # Extract noise for each channel
    noise_b = noise[:, :, 0].flatten()
    noise_g = noise[:, :, 1].flatten()
    noise_r = noise[:, :, 2].flatten()
    noise_combined = noise.flatten()
    
    # Calculate statistics
    mean_noise = np.mean(noise_combined)
    std_noise = np.std(noise_combined)
    
    print(f"\nNoise Statistics:")
    print(f"  Mean: {mean_noise:.4f}")
    print(f"  Std Dev: {std_noise:.4f}")
    print(f"  Min: {np.min(noise_combined):.4f}")
    print(f"  Max: {np.max(noise_combined):.4f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ===== Row 1: Original, Noisy, Noise visualization =====
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Noisy Image', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    # Visualize noise (normalize for display)
    noise_vis = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    noise_vis = noise_vis.astype(np.uint8)
    ax3.imshow(cv2.cvtColor(noise_vis, cv2.COLOR_BGR2RGB))
    ax3.set_title('Noise Visualization (Normalized)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    # Absolute difference
    diff = cv2.absdiff(original_img, noisy_img)
    ax4.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    ax4.set_title('Absolute Difference', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # ===== Row 2: Per-channel noise histograms =====
    colors = ['blue', 'green', 'red']
    noise_channels = [noise_b, noise_g, noise_r]
    channel_names = ['Blue Channel', 'Green Channel', 'Red Channel']
    
    for i, (noise_ch, color, name) in enumerate(zip(noise_channels, colors, channel_names)):
        ax = plt.subplot(3, 4, 5 + i)
        
        # Plot histogram
        counts, bins, patches = ax.hist(noise_ch, bins=100, color=color, alpha=0.7, 
                                       edgecolor='black', linewidth=0.5)
        
        # Overlay Gaussian fit
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
    
    # ===== Combined noise histogram =====
    ax8 = plt.subplot(3, 4, 8)
    counts, bins, patches = ax8.hist(noise_combined, bins=100, color='gray', alpha=0.7, 
                                    edgecolor='black', linewidth=0.5)
    
    # Overlay Gaussian fit
    mu_all = np.mean(noise_combined)
    sigma_all = np.std(noise_combined)
    x = np.linspace(noise_combined.min(), noise_combined.max(), 100)
    gaussian_fit = len(noise_combined) * (bins[1] - bins[0]) * \
                  (1 / (sigma_all * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x - mu_all) / sigma_all) ** 2)
    ax8.plot(x, gaussian_fit, 'r--', linewidth=2, 
            label=f'Gaussian Fit\nμ={mu_all:.2f}, σ={sigma_all:.2f}')
    
    ax8.set_title('Combined Noise Distribution (All Channels)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Noise Value', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.grid(alpha=0.3)
    ax8.legend(fontsize=8)
    ax8.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # ===== Row 3: Original vs Noisy intensity histograms =====
    # Original image histograms
    ax9 = plt.subplot(3, 4, 9)
    for i, color in enumerate(['b', 'g', 'r']):
        ax9.hist(original_img[:, :, i].flatten(), bins=50, color=color, 
                alpha=0.5, label=f'{["Blue", "Green", "Red"][i]}')
    ax9.set_title('Original Image - RGB Histograms', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Pixel Intensity', fontsize=10)
    ax9.set_ylabel('Frequency', fontsize=10)
    ax9.set_xlim([0, 255])
    ax9.grid(alpha=0.3)
    ax9.legend(fontsize=8)
    
    # Noisy image histograms
    ax10 = plt.subplot(3, 4, 10)
    for i, color in enumerate(['b', 'g', 'r']):
        ax10.hist(noisy_img[:, :, i].flatten(), bins=50, color=color, 
                 alpha=0.5, label=f'{["Blue", "Green", "Red"][i]}')
    ax10.set_title('Noisy Image - RGB Histograms', fontsize=11, fontweight='bold')
    ax10.set_xlabel('Pixel Intensity', fontsize=10)
    ax10.set_ylabel('Frequency', fontsize=10)
    ax10.set_xlim([0, 255])
    ax10.grid(alpha=0.3)
    ax10.legend(fontsize=8)
    
    # Q-Q plot to verify Gaussian distribution
    ax11 = plt.subplot(3, 4, 11)
    from scipy import stats
    stats.probplot(noise_combined[::100], dist="norm", plot=ax11)  # Sample every 100th point for speed
    ax11.set_title('Q-Q Plot (Normality Test)', fontsize=11, fontweight='bold')
    ax11.grid(alpha=0.3)
    
    # Noise statistics summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    stats_text = f"""
    NOISE STATISTICS SUMMARY
    {'='*30}
    
    Combined (All Channels):
      Mean: {mu_all:.4f}
      Std Dev: {sigma_all:.4f}
      Min: {np.min(noise_combined):.4f}
      Max: {np.max(noise_combined):.4f}
    
    Per Channel:
      Blue:
        μ = {np.mean(noise_b):.4f}
        σ = {np.std(noise_b):.4f}
      Green:
        μ = {np.mean(noise_g):.4f}
        σ = {np.std(noise_g):.4f}
      Red:
        μ = {np.mean(noise_r):.4f}
        σ = {np.std(noise_r):.4f}
    
    Expected: μ ≈ 0, σ ≈ {std_noise:.2f}
    """
    ax12.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, noise

def analyze_multiple_images(original_dir, noisy_dir, num_samples=3):
    """
    Analyze multiple image pairs to get overall noise statistics.
    
    Parameters:
    -----------
    original_dir : Path or str
        Directory containing original images
    noisy_dir : Path or str
        Directory containing noisy images
    num_samples : int
        Number of samples to analyze in detail
    """
    original_dir = Path(original_dir)
    noisy_dir = Path(noisy_dir)
    
    # Get list of images
    original_images = sorted(list(original_dir.glob("*.png")) + 
                           list(original_dir.glob("*.jpg")))
    
    if not original_images:
        print(f"No images found in {original_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing Gaussian Noise Distribution")
    print(f"{'='*60}")
    print(f"Original images directory: {original_dir}")
    print(f"Noisy images directory: {noisy_dir}")
    print(f"Total images found: {len(original_images)}")
    print(f"Analyzing {min(num_samples, len(original_images))} samples in detail...")
    
    # Analyze samples in detail
    for i, original_path in enumerate(original_images[:num_samples]):
        noisy_path = noisy_dir / f"noisy_{original_path.name}"
        
        if not noisy_path.exists():
            print(f"\nSkipping {original_path.name} - noisy version not found")
            continue
        
        print(f"\n--- Sample {i+1}: {original_path.name} ---")
        
        original_img, noisy_img = load_image_pair(original_path, noisy_path)
        
        if original_img is None or noisy_img is None:
            print(f"Error loading images for {original_path.name}")
            continue
        
        fig, noise = plot_gaussian_noise_distribution(
            original_img, noisy_img, 
            title=f"Gaussian Noise Analysis - {original_path.name}"
        )
        
        plt.show()
        
        # Ask if user wants to continue
        if i < num_samples - 1:
            response = input("\nPress Enter to continue to next sample (or 'q' to quit): ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    # Configuration
    ORIGINAL_DIR = Path("output_images")
    NOISY_DIR = Path("gaussian_noise_10_sigma")
    NUM_SAMPLES = 3
    
    # Run analysis
    analyze_multiple_images(ORIGINAL_DIR, NOISY_DIR, num_samples=NUM_SAMPLES)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
