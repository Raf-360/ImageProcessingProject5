import os
import cv2
import numpy as np
import csv
import random

# ==============================
# Adjustable Parameters
# ==============================
input_folder = r"C:\\Users\\rafae\\OneDrive - Texas Tech University\\Fall 2025\\Image Processing\\Project 5\\output_images"
output_folder = r"C:\\Users\\rafae\\OneDrive - Texas Tech University\\Fall 2025\\Image Processing\\Project 5\\Noise_Images_Salt_And_Pepper_10_Percent"
noise_density = 0.1  # % of total pixels affected (0.02 = 2%)
# ==============================

# --- Create output directory ---
os.makedirs(output_folder, exist_ok=True)

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

# --- Process each image in input folder ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)

        # --- Apply noise ---
        noisy_image, salt_coords, pepper_coords = add_salt_pepper_noise(image, noise_density)

        # --- Save noisy image ---
        output_image_path = os.path.join(output_folder, f"noisy_{filename}")
        cv2.imwrite(output_image_path, noisy_image)

        # --- Save pixel coordinates to CSV ---
        csv_filename = f"noisy_{os.path.splitext(filename)[0]}.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "X", "Y"])
            for x, y in salt_coords:
                writer.writerow(["Salt", x, y])
            for x, y in pepper_coords:
                writer.writerow(["Pepper", x, y])

        print(f"✅ Processed {filename}: {len(salt_coords)} salt + {len(pepper_coords)} pepper pixels")

print("\nAll images processed and saved to:", output_folder)
