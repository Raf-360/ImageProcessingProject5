"""
Image I/O utilities for loading and saving images.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_images(image_path: Path) -> List[np.ndarray]:
    """
    Load all image files from a directory or a single image file.
    
    Args:
        image_path: Directory containing images or single image file
        
    Returns:
        List of loaded images (BGR format)
    """
    loaded_images = []
    
    # Check if it's a single file
    if image_path.is_file():
        img = cv.imread(str(image_path))
        if img is not None:
            loaded_images.append(img)
        return loaded_images
    
    # Otherwise treat as directory
    if image_path.exists():
        # Support multiple image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_path.rglob(ext))
        
        # Sort for consistent ordering
        image_files = sorted(image_files)
        
        for img_path in image_files:
            img = cv.imread(str(img_path))
            if img is not None:
                loaded_images.append(img)
    return loaded_images


def load_image_pair(noisy_path: Path, clean_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load pairs of noisy and clean images.
    
    Args:
        noisy_path: Directory or file containing noisy image(s)
        clean_path: Directory or file containing clean image(s)
        
    Returns:
        Tuple of (noisy_images, clean_images)
    """
    noisy_images = load_images(noisy_path)
    clean_images = load_images(clean_path)
    
    # Ensure we have matching pairs
    if len(noisy_images) != len(clean_images):
        print(f"⚠️  Warning: Mismatch in image counts - noisy: {len(noisy_images)}, clean: {len(clean_images)}")
    
    return noisy_images, clean_images


def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Save image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), image)
    print(f"✓ Saved to {output_path}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to float32 [0, 1] range.
    
    Args:
        image: Input image (uint8)
        
    Returns:
        Normalized image (float32)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image back to uint8 [0, 255] range.
    
    Args:
        image: Normalized image (float32)
        
    Returns:
        Denormalized image (uint8)
    """
    return np.clip(image * 255.0, 0, 255.0).astype(np.uint8)
