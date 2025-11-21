import os
import cv2
import numpy as np
import csv
import random
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
                Examples:
                %(prog)s -i clean_images -o noisy_output -s 10
                %(prog)s --input images/ --output output/ --sigma 25
                """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input folder containing clean images"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output folder for noisy images"
    )
    
    parser.add_argument(
        "-s", "--sigma",
        type=float,
        default=10.0,
        help="Standard deviation of Gaussian noise (default: 10.0)"
    )
    
    parser.add_argument(
        "-d", "--density",
        type=float,
        default=0.1,
        help="Noise density for salt & pepper (0.1 = 10%%, default: 0.1)"
    )
    
    return parser.parse_args()


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


def main():
    """Main function to process images."""
    args = parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    sigma = args.sigma
    
    # Validate input folder exists and is a directory
    if not input_folder.is_dir():
        print(f"❌ Error: Input folder '{input_folder}' does not exist or is not a directory")
        return 1
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process each image in input folder
    processed_count = 0
    for image_file in input_folder.glob("*"):
        if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            image = cv2.imread(str(image_file))
            
            if image is None:
                print(f"⚠️  Warning: Could not read {image_file.name}, skipping")
                continue

            # Apply Gaussian noise
            noisy_image = add_gaussian_noise(image, sigma)

            # Save noisy image
            output_image_path = output_folder / f"noisy_{image_file.name}"
            cv2.imwrite(str(output_image_path), noisy_image)

            print(f"✅ Processed {image_file.name}: Added Gaussian noise (sigma={sigma})")
            processed_count += 1

    if processed_count == 0:
        print(f"\n⚠️  No images found in {input_folder}")
        return 1
    
    print(f"\n✅ Successfully processed {processed_count} image(s)")
    print(f"📁 Output saved to: {output_folder}")
    return 0


if __name__ == "__main__":
    exit(main())
