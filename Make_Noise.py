import os
import cv2
import numpy as np
import multiprocessing as mp
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from functools import partial
import random
import csv

# =====================================================
# SETTINGS
# =====================================================
NOISE_DENSITY = 0.10     # Salt & Pepper density
SIGMA = 25               # Gaussian noise sigma (if enabled later)


# =====================================================
# SALT & PEPPER NOISE
# =====================================================
def add_sp_noise(image, density=0.10):
    """Add salt & pepper noise and return (noisy, mask)."""
    h, w, c = image.shape
    total_pixels = h * w
    num = int(total_pixels * density)

    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    salt_mask = np.random.rand(num) < 0.5

    noisy = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    noisy[ys[salt_mask], xs[salt_mask]] = [255, 255, 255]  # salt
    noisy[ys[~salt_mask], xs[~salt_mask]] = [0, 0, 0]      # pepper

    mask[ys, xs] = 255  # mark noisy pixel

    return noisy, mask


# =====================================================
# MULTIPROCESS WORKER (S&P)
# =====================================================
def process_one(clean_path, output_folder, noise_density):
    clean_path = Path(clean_path)

    image = cv2.imread(str(clean_path))
    if image is None:
        return f"[ERROR] Could not read: {clean_path.name}"

    noisy, mask = add_sp_noise(image, density=noise_density)

    noisy_name = f"noisy_{clean_path.name}"
    mask_name = f"noisy_{clean_path.stem}_mask.png"

    cv2.imwrite(str(output_folder / noisy_name), noisy)
    cv2.imwrite(str(output_folder / mask_name), mask)

    return f"Processed {clean_path.name}"


# =====================================================
# OPTIONAL — GAUSSIAN NOISE (COMMENTED OUT FOR NOW)
# =====================================================
"""
def add_gaussian_noise(image, sigma):
    # Add Gaussian noise
    noisy = image.copy().astype(np.float32)
    h, w, c = image.shape
    gaussian = np.random.normal(0, sigma, (h, w, c))
    noisy = noisy + gaussian
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def gaussian_generator(input_folder, output_folder):
    # Make output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob("*"):
        if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            image = cv2.imread(str(image_file))
            noisy = add_gaussian_noise(image, SIGMA)

            out_path = output_folder / f"noisy_{image_file.name}"
            cv2.imwrite(str(out_path), noisy)
            print(f"Added Gaussian noise: {image_file.name}")
"""


# =====================================================
# MAIN PROGRAM (GUI + multiprocessing for S&P)
# =====================================================
if __name__ == "__main__":
    # GUI Folder Selection
    root = tk.Tk()
    root.withdraw()

    print("\nSelect CLEAN image folder:")
    clean_dir = Path(filedialog.askdirectory())

    print("\nSelect OUTPUT folder:")
    output_folder = Path(filedialog.askdirectory())
    output_folder.mkdir(exist_ok=True, parents=True)

    # Gather images
    clean_paths = sorted(list(clean_dir.glob("*.png")))
    total = len(clean_paths)

    print(f"\nFound {total} clean images.")
    print(f"Starting Salt & Pepper noise generation (density={NOISE_DENSITY*100:.0f}%)...")
    print()

    # SALT & PEPPER MULTIPROCESSING
    cpu_cores = mp.cpu_count()
    print(f"Using {cpu_cores} CPU cores.\n")

    worker = partial(process_one,
                     output_folder=output_folder,
                     noise_density=NOISE_DENSITY)

    with mp.Pool(cpu_cores) as pool:
        for result in pool.imap_unordered(worker, clean_paths):
            print(result)

    print("\n🎉 Done! All Salt & Pepper noisy images + masks generated.")

    # =============================================================
    # OPTIONAL GAUSSIAN GENERATOR (DISABLED)
    # Enable manually by removing the triple quote block above
    # =============================================================
    """
    print("\nRunning Gaussian noise generator...")
    gaussian_generator(clean_dir, output_folder)
    print("\n🎉 Gaussian noise generation complete!")
    """
