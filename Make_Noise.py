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
        description="Add noise to images (Gaussian or Salt & Pepper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
                Examples:
                %(prog)s -i clean_images -o noisy_output -t gaussian -s 10
                %(prog)s -i clean_images -o noisy_output -t salt_pepper -d 0.05
                %(prog)s --input images/ --output output/ --type gaussian --sigma 25
                """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input folder or single image file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output folder for noisy images"
    )
    
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["gaussian", "salt_pepper"],
        default="gaussian",
        help="Type of noise to add: gaussian or salt_pepper (default: gaussian)"
    )
    
    parser.add_argument(
        "-s", "--sigma",
        type=float,
        default=10.0,
        help="Standard deviation of Gaussian noise, must be > 0 (default: 10.0)"
    )
    
    parser.add_argument(
        "-d", "--density",
        type=float,
        default=0.1,
        help="Noise density for salt & pepper, must be between 0 and 1 (default: 0.1)"
    )
    
    return parser.parse_args()


def validate_parameters(noise_type, sigma, density):
    """Validate noise parameters."""
    if noise_type == "gaussian":
        if sigma <= 0:
            raise ValueError(f"Sigma must be greater than 0, got {sigma}")
        if sigma > 255:
            print(f"⚠️  Warning: Very large sigma value ({sigma}) may result in completely noisy images")
    
    elif noise_type == "salt_pepper":
        if not (0 <= density <= 1):
            raise ValueError(f"Density must be between 0 and 1, got {density}")
        if density > 0.5:
            print(f"⚠️  Warning: High density value ({density}) will heavily corrupt images")


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
    noise_type = args.type
    sigma = args.sigma
    density = args.density
    
    # Validate parameters
    try:
        validate_parameters(noise_type, sigma, density)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Validate input folder exists
    if not input_folder.exists():
        print(f"❌ Error: Input path '{input_folder}' does not exist")
        return 1
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Check if input is a single file or directory
    if input_folder.is_file():
        # Process single image
        if input_folder.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
            print(f"❌ Error: Unsupported image format: {input_folder.suffix}")
            return 1
        
        image_files = [input_folder]
    else:
        # Process all images in directory
        image_files = [f for f in input_folder.glob("*") 
                      if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
    
    if not image_files:
        print(f"❌ Error: No valid images found")
        return 1
    
    # Process each image
    processed_count = 0
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        
        if image is None:
            print(f"⚠️  Warning: Could not read {image_file.name}, skipping")
            continue

        # Apply noise based on type
        if noise_type == "gaussian":
            noisy_image = add_gaussian_noise(image, sigma)
            noise_desc = f"Gaussian (sigma={sigma})"
        else:  # salt_pepper
            noisy_image, _, _ = add_salt_pepper_noise(image, density)
            noise_desc = f"Salt & Pepper (density={density})"

        # Save noisy image
        output_image_path = output_folder / f"noisy_{image_file.name}"
        cv2.imwrite(str(output_image_path), noisy_image)

        print(f"✅ Processed {image_file.name}: Added {noise_desc}")
        processed_count += 1

    if processed_count == 0:
        print(f"\n⚠️  Failed to process any images")
        return 1
    
    print(f"\n✅ Successfully processed {processed_count} image(s)")
    print(f"📁 Output saved to: {output_folder}")
    return 0


if __name__ == "__main__":
    exit(main())
