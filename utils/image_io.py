"""
Image I/O utilities for loading and saving images.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_images(image_path: Path) -> List[np.ndarray]:
    """
    Load all image files from a directory (supports png, jpg, jpeg).
    
    Args:
        image_path: Directory containing images
        
    Returns:
        List of loaded images (BGR format)
    """
    loaded_images = []
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
        noisy_path: Directory containing noisy images
        clean_path: Directory containing clean images
        
    Returns:
        Tuple of (noisy_images, clean_images)
    """
    noisy_images = load_images(noisy_path)
    clean_images = load_images(clean_path)
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
