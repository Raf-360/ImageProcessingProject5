import os
import cv2
import numpy as np
import csv
import random
from pathlib import Path

# ==============================
# Adjustable Parameters
# ==============================
input_folder = Path("clean_images")
output_folder = Path("gaussian_noise_25_sigma")
noise_density = 0.1  # % of total pixels affected (0.02 = 2%)
SIGMA = 25
# ==============================

# --- Create output directory ---
output_folder.mkdir(parents=True, exist_ok=True)

# --- Helper: add salt & pepper noise and track positions ---
def add_salt_pepper_noise(image, density):
    noisy = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    num_noisy = int(total_pixels * density)

    coords_salt = []
    coords_pepper = []

    for _ in range(num_noisy):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if random.random() < 0.5:
            noisy[y, x] = [255, 255, 255]  # salt (white)
            coords_salt.append((x, y))
        else:
            noisy[y, x] = [0, 0, 0]  # pepper (black)
            coords_pepper.append((x, y))

    return noisy, coords_salt, coords_pepper


def add_gaussian_noise(image, sigma):
    """
    Add Gaussian noise to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR format)
    sigma : float
        Standard deviation of the Gaussian noise
        
    Returns:
    --------
    noisy : numpy.ndarray
        Image with added Gaussian noise
    """
    noisy = image.copy().astype(np.float32)
    
    # Generate Gaussian noise with mean=0 and std=sigma
    h, w, c = image.shape
    gaussian_noise = np.random.normal(0, sigma, (h, w, c))
    
    # Add noise to image
    noisy = noisy + gaussian_noise
    
    # Clip values to valid range [0, 255] and convert back to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

# --- Process each image in input folder ---
for image_file in input_folder.glob("*"):
    if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        image = cv2.imread(str(image_file))

        # --- Apply Gaussian noise ---
        noisy_image = add_gaussian_noise(image, SIGMA)

        # --- Save noisy image ---
        output_image_path = output_folder / f"noisy_{image_file.name}"
        cv2.imwrite(str(output_image_path), noisy_image)

        print(f"✅ Processed {image_file.name}: Added Gaussian noise (sigma={SIGMA})")

print("\nAll images processed and saved to:", output_folder)
