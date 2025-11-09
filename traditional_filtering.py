import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path

def load_noise_coordinates(csv_path):
    """Load salt and pepper noise coordinates from CSV file."""
    salt_coords = []
    pepper_coords = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = int(row['X']), int(row['Y'])
            if row['Type'] == 'Salt':
                salt_coords.append((x, y))
            else:
                pepper_coords.append((x, y))
    
    return salt_coords, pepper_coords

def plot_histograms(original_img, noisy_img, title):
    """Plot RGB and grayscale histograms for original and noisy images."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=16)
    
    # Convert to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    noisy_gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
    
    colors = ['b', 'g', 'r']
    channel_names = ['Blue', 'Green', 'Red']
    
    # Plot RGB histograms for original image
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        axes[0, i].hist(original_img[:,:,i].ravel(), bins=256, range=[0,256], 
                        color=color, alpha=0.7)
        axes[0, i].set_title(f'Original - {name} Channel')
        axes[0, i].set_xlim([0, 256])
        axes[0, i].set_xlabel('Pixel Intensity')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(alpha=0.3)
    
    # Plot grayscale histogram for original
    axes[0, 3].hist(original_gray.ravel(), bins=256, range=[0,256], 
                    color='gray', alpha=0.7)
    axes[0, 3].set_title('Original - Grayscale')
    axes[0, 3].set_xlim([0, 256])
    axes[0, 3].set_xlabel('Pixel Intensity')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].grid(alpha=0.3)
    
    # Plot RGB histograms for noisy image
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        axes[1, i].hist(noisy_img[:,:,i].ravel(), bins=256, range=[0,256], 
                        color=color, alpha=0.7)
        axes[1, i].set_title(f'Noisy - {name} Channel')
        axes[1, i].set_xlim([0, 256])
        axes[1, i].set_xlabel('Pixel Intensity')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(alpha=0.3)
        
        # Highlight spikes at 0 and 255
        axes[1, i].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Pepper (0)')
        axes[1, i].axvline(x=255, color='white', linestyle='--', linewidth=2, label='Salt (255)')
        axes[1, i].legend()
    
    # Plot grayscale histogram for noisy
    axes[1, 3].hist(noisy_gray.ravel(), bins=256, range=[0,256], 
                    color='gray', alpha=0.7)
    axes[1, 3].set_title('Noisy - Grayscale')
    axes[1, 3].set_xlim([0, 256])
    axes[1, 3].set_xlabel('Pixel Intensity')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].grid(alpha=0.3)
    axes[1, 3].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Pepper (0)')
    axes[1, 3].axvline(x=255, color='red', linestyle='--', linewidth=2, label='Salt (255)')
    axes[1, 3].legend()
    
    plt.tight_layout()
    return fig

def plot_noise_distribution(salt_coords, pepper_coords, img_shape, title):
    """Plot spatial distribution of noise pixels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    
    # Combined noise distribution
    if salt_coords:
        salt_x, salt_y = zip(*salt_coords)
        axes[0].scatter(salt_x, salt_y, c='white', s=1, marker='s', 
                       edgecolors='red', linewidths=0.5, label=f'Salt ({len(salt_coords)})')
    if pepper_coords:
        pepper_x, pepper_y = zip(*pepper_coords)
        axes[0].scatter(pepper_x, pepper_y, c='black', s=1, marker='s', 
                       edgecolors='blue', linewidths=0.5, label=f'Pepper ({len(pepper_coords)})')
    
    axes[0].set_xlim([0, img_shape[1]])
    axes[0].set_ylim([img_shape[0], 0])
    axes[0].set_title('Combined Noise Distribution')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Salt noise only
    if salt_coords:
        axes[1].scatter(salt_x, salt_y, c='white', s=2, marker='s', 
                       edgecolors='red', linewidths=0.5)
    axes[1].set_xlim([0, img_shape[1]])
    axes[1].set_ylim([img_shape[0], 0])
    axes[1].set_title(f'Salt Noise Only ({len(salt_coords)} pixels)')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].grid(alpha=0.3)
    
    # Pepper noise only
    if pepper_coords:
        axes[2].scatter(pepper_x, pepper_y, c='black', s=2, marker='s')
    axes[2].set_xlim([0, img_shape[1]])
    axes[2].set_ylim([img_shape[0], 0])
    axes[2].set_title(f'Pepper Noise Only ({len(pepper_coords)} pixels)')
    axes[2].set_xlabel('X Coordinate')
    axes[2].set_ylabel('Y Coordinate')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_sample(noise_level='2_Percent'):
    """Visualize a sample image with its noise characteristics."""
    
    # Find directories
    if noise_level == '2_Percent':
        noise_dir = 'Noise_Images_Salt_And_Pepper_2_Percent'
    else:
        noise_dir = 'Noise_Images_Salt_And_Pepper_10_Percent'
    
    original_dir = 'Generated_Images'
    
    # Get a sample image
    csv_files = list(Path(noise_dir).glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {noise_dir}")
        return
    
    # Load first sample
    csv_path = csv_files[0]
    print(f"Analyzing: {csv_path.name}")
    
    # Extract base filename (remove 'noisy_' prefix and '.csv' suffix)
    base_name = csv_path.stem.replace('noisy_', '') + '.png'
    
    # Load images
    original_path = os.path.join(original_dir, base_name)
    noisy_path = os.path.join(noise_dir, csv_path.stem + '.png')
    
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return
    
    original_img = cv2.imread(original_path)
    noisy_img = cv2.imread(noisy_path)
    
    # Load noise coordinates
    salt_coords, pepper_coords = load_noise_coordinates(str(csv_path))
    
    # Calculate statistics
    total_pixels = original_img.shape[0] * original_img.shape[1]
    noise_pixels = len(salt_coords) + len(pepper_coords)
    actual_noise_percent = (noise_pixels / total_pixels) * 100
    
    print(f"\nNoise Statistics:")
    print(f"  Image size: {original_img.shape[1]}x{original_img.shape[0]} ({total_pixels} pixels)")
    print(f"  Salt pixels: {len(salt_coords)}")
    print(f"  Pepper pixels: {len(pepper_coords)}")
    print(f"  Total noise pixels: {noise_pixels}")
    print(f"  Actual noise percentage: {actual_noise_percent:.2f}%")
    
    # Create visualizations
    print("\nGenerating histograms...")
    hist_fig = plot_histograms(original_img, noisy_img, 
                               f'Histogram Analysis - {noise_level.replace("_", " ")} Noise')
    
    print("Generating spatial distribution plot...")
    dist_fig = plot_noise_distribution(salt_coords, pepper_coords, original_img.shape,
                                       f'Noise Spatial Distribution - {noise_level.replace("_", " ")}')
    
    # Show side-by-side images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Noisy Image ({actual_noise_percent:.2f}% noise)')
    axes[1].axis('off')
    
    # Difference image
    diff = cv2.absdiff(original_img, noisy_img)
    axes[2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    print("=== Visualizing 2% Noise Level ===")
    visualize_sample('2_Percent')
    
    print("\n" + "="*50 + "\n")
    print("=== Visualizing 10% Noise Level ===")
    visualize_sample('10_Percent')