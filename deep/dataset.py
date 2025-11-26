import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2 as cv
from typing import Tuple, List, Optional
import random

# Test with DataLoader
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class DenoisingDataset(Dataset):
    def __init__(
        self,
        data_root: str, 
        split: str = "train",
        categories: List[str] = ["synthetic", "xray", "jellyfish"],
        noise_levels: List[int] = [15, 25, 55], 
        patch_size: int = 40,  
        stride: int = 40,  
        augment: bool = True,
        grayscale: bool = True,
        max_patches_per_image: int = 20,  # ✅ Reduced to 20
        max_images: int = 100,  # ✅ START WITH ONLY 100 IMAGES FOR TESTING!
        lazy_load: bool = False  # ✅ Set True to load patches on-the-fly (saves memory)
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.categories = categories
        self.noise_levels = noise_levels if isinstance(noise_levels, list) else [noise_levels]
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches_per_image = max_patches_per_image
        self.max_images = max_images
        self.augment = augment and (split == 'train')
        self.grayscale = grayscale
        
        # Storage for patches
        self.clean_patches = []
        self.noisy_patches = []
        self.noise_levels_per_patch = []  # Track which noise level each patch has
        
        # Load and extract patches
        self._load_dataset()       
        
        print(f"\n{split.upper()} Dataset loaded:")
        print(f"  Categories: {categories}")
        print(f"  Noise levels: σ={noise_levels}")
        if self.lazy_load:
            print(f"  Total image pairs: {len(self.image_pairs)}")
            print(f"  Patches per image: {self.max_patches_per_image} (loaded on-the-fly)")
            print(f"  Effective patches: ~{len(self.image_pairs) * self.max_patches_per_image}")
        else:
            print(f"  Total patches: {len(self.clean_patches)}")
        print(f"  Patch size: {patch_size}×{patch_size}")
        print(f"  Augmentation: {self.augment}")
        
        
    def _load_dataset(self):
        """Load all images and extract patches."""
        split_path = self.data_root / self.split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found: {split_path}")
        
        for category in self.categories:
            category_path = split_path / category
            
            if not category_path.exists():
                print(f"Warning: Category {category} not found, skipping...")
                continue
            
            # Loop through all noise levels
            for noise_level in self.noise_levels:
                # Paths to clean and noisy folders
                clean_folder = category_path / 'clean'
                
                # Try multiple possible folder name patterns
                possible_noisy_folders = [
                    category_path / f'gaussian_noise_{noise_level}_sigma',
                    category_path / f'XRAY_gaussian_noise_{noise_level}_sigma',
                    category_path / f'{category}_gaussian_noise_{noise_level}_sigma',
                ]
                
                # Find which folder exists
                noisy_folder = None
                for folder in possible_noisy_folders:
                    if folder.exists():
                        noisy_folder = folder
                        break
                
                # Also try glob pattern in case of weird naming
                if noisy_folder is None:
                    glob_matches = list(category_path.glob(f'*{noise_level}*sigma*'))
                    if glob_matches:
                        noisy_folder = glob_matches[0]
                
                if not clean_folder.exists():
                    print(f"  Warning: Clean folder not found: {clean_folder}")
                    continue
                    
                if noisy_folder is None or not noisy_folder.exists():
                    print(f"  Warning: Noisy folder for {category} σ={noise_level} not found")
                    print(f"           Searched in: {category_path}")
                    continue
                
                # Get all image files
                clean_images = sorted(self._get_image_files(clean_folder))
                noisy_images = sorted(self._get_image_files(noisy_folder))
                
                print(f"    Found {len(clean_images)} clean, {len(noisy_images)} noisy images")
                
                # Match clean and noisy pairs by filename
                matched_pairs = self._match_image_pairs(clean_images, noisy_images)
                
                # ✅ Limit number of images to prevent memory explosion
                if self.max_images:
                    matched_pairs = matched_pairs[:self.max_images]
                
                print(f"  Loading {self.split}/{category} (σ={noise_level}): {len(matched_pairs)} image pairs from {noisy_folder.name}")
                
                if len(matched_pairs) > 0:
                    print(f"    Example pair: {matched_pairs[0][0].name} <-> {matched_pairs[0][1].name}")
                
                # Extract patches from each pair
                patches_before = len(self.clean_patches) if not self.lazy_load else len(self.image_pairs)
                for clean_path, noisy_path in matched_pairs:
                    if self.lazy_load:
                        # Just store paths
                        self.image_pairs.append((clean_path, noisy_path, noise_level))
                    else:
                        # Extract patches immediately
                        self._extract_patches_from_pair(clean_path, noisy_path, noise_level)
                patches_after = len(self.clean_patches) if not self.lazy_load else len(self.image_pairs)
                
                print(f"    Extracted {patches_after - patches_before} {'image pairs' if self.lazy_load else 'patches'}")    
                    
                    
    def _extract_patches_from_pair(self, clean_path: Path, noisy_path: Path, noise_level: int):
        """Extract matching patches from a clean/noisy image pair."""
        # Load images
        clean_img = cv.imread(str(clean_path), cv.IMREAD_UNCHANGED)
        noisy_img = cv.imread(str(noisy_path), cv.IMREAD_UNCHANGED)
        
        if clean_img is None or noisy_img is None:
            print(f"Warning: Failed to load {clean_path.name}")
            return
        
        # Convert to grayscale if needed
        if self.grayscale and len(clean_img.shape) == 3:
            clean_img = cv.cvtColor(clean_img, cv.COLOR_BGR2GRAY)
            noisy_img = cv.cvtColor(noisy_img, cv.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        clean_img = clean_img.astype(np.float32) / 255.0
        noisy_img = noisy_img.astype(np.float32) / 255.0
        
        # Extract patches
        h, w = clean_img.shape[:2]
        
        # ✅ Use random sampling for training, grid for validation/test
        if self.split == 'train' and self.max_patches_per_image:
            # Random sampling - prevents memory issues
            for _ in range(min(self.max_patches_per_image, (h * w) // (self.patch_size ** 2))):
                # Random top-left corner
                i = random.randint(0, max(0, h - self.patch_size))
                j = random.randint(0, max(0, w - self.patch_size))
                
                clean_patch = clean_img[i:i+self.patch_size, j:j+self.patch_size]
                noisy_patch = noisy_img[i:i+self.patch_size, j:j+self.patch_size]
                
                # Store patches
                self.clean_patches.append(clean_patch)
                self.noisy_patches.append(noisy_patch)
                self.noise_levels_per_patch.append(noise_level)
        else:
            # Grid sampling for validation/test (deterministic)
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    clean_patch = clean_img[i:i+self.patch_size, j:j+self.patch_size]
                    noisy_patch = noisy_img[i:i+self.patch_size, j:j+self.patch_size]
                    
                    # Store patches
                    self.clean_patches.append(clean_patch)
                    self.noisy_patches.append(noisy_patch)
                    self.noise_levels_per_patch.append(noise_level) 
                
    
    def _get_image_files(self, folder: Path) -> List[Path]:
        extensions = {".png", ".jpeg", ".jpg"}
        
        return [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    
    def _match_image_pairs(self, clean_images: List[Path], noisy_images: List[Path]) -> List[Tuple[Path, Path]]:
        """Match clean and noisy images by filename, handling 'noisy_' prefix."""
        # Create dict with clean filenames as keys
        clean_dict = {img.name: img for img in clean_images}
        
        # Create dict for noisy, removing 'noisy_' prefix if present
        noisy_dict = {}
        for img in noisy_images:
            # Remove 'noisy_' prefix if it exists
            clean_name = img.name.replace('noisy_', '', 1)
            noisy_dict[clean_name] = img
        
        # Find common filenames
        common_names = set(clean_dict.keys()) & set(noisy_dict.keys())
        
        matched = [(clean_dict[name], noisy_dict[name]) for name in sorted(common_names)]
        
        if len(matched) == 0 and len(clean_images) > 0 and len(noisy_images) > 0:
            print(f"    DEBUG: No matches found!")
            print(f"      Clean example: {clean_images[0].name}")
            print(f"      Noisy example: {noisy_images[0].name}")
        
        return matched
    
    def __len__(self) -> int:
        """Return total number of patches."""
        if self.lazy_load:
            return len(self.image_pairs) * self.max_patches_per_image
        else:
            return len(self.clean_patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single clean/noisy patch pair.
        
        Args:
            idx: Index of the patch
        
        Returns:
            Tuple of (noisy_patch, clean_patch) as torch tensors
            Shape: (C, H, W) where C=1 for grayscale, C=3 for RGB
        """
        if self.lazy_load:
            # Load patch on-the-fly
            img_idx = idx // self.max_patches_per_image
            patch_idx = idx % self.max_patches_per_image
            
            clean_path, noisy_path, noise_level = self.image_pairs[img_idx]
            
            # Load full images
            clean_img = cv.imread(str(clean_path), cv.IMREAD_UNCHANGED)
            noisy_img = cv.imread(str(noisy_path), cv.IMREAD_UNCHANGED)
            
            # Convert to grayscale if needed
            if self.grayscale and len(clean_img.shape) == 3:
                clean_img = cv.cvtColor(clean_img, cv.COLOR_BGR2GRAY)
                noisy_img = cv.cvtColor(noisy_img, cv.COLOR_BGR2GRAY)
            
            # Normalize
            clean_img = clean_img.astype(np.float32) / 255.0
            noisy_img = noisy_img.astype(np.float32) / 255.0
            
            # Extract random patch
            h, w = clean_img.shape[:2]
            i = random.randint(0, max(0, h - self.patch_size))
            j = random.randint(0, max(0, w - self.patch_size))
            
            clean_patch = clean_img[i:i+self.patch_size, j:j+self.patch_size]
            noisy_patch = noisy_img[i:i+self.patch_size, j:j+self.patch_size]
        else:
            # Get pre-extracted patch
            clean_patch = self.clean_patches[idx].copy()
            noisy_patch = self.noisy_patches[idx].copy()
        
        # Apply augmentation if enabled
        if self.augment:
            clean_patch, noisy_patch = self._augment(clean_patch, noisy_patch)
        
        # Add channel dimension for grayscale: (H, W) -> (1, H, W)
        if len(clean_patch.shape) == 2:
            clean_patch = clean_patch[np.newaxis, ...]
            noisy_patch = noisy_patch[np.newaxis, ...]
        else:
            # RGB: (H, W, C) -> (C, H, W)
            clean_patch = np.transpose(clean_patch, (2, 0, 1))
            noisy_patch = np.transpose(noisy_patch, (2, 0, 1))
        
        # Convert to torch tensors
        clean_tensor = torch.from_numpy(clean_patch).float()
        noisy_tensor = torch.from_numpy(noisy_patch).float()
        
        return noisy_tensor, clean_tensor
    
    
    def _augment(self, clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to patch pairs.
        
        Augmentations:
        - Random horizontal flip (50% chance)
        - Random vertical flip (50% chance)
        - Random 90° rotation (0°, 90°, 180°, 270°)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)
        
        # Random vertical flip
        if random.random() > 0.5:
            clean = np.flipud(clean)
            noisy = np.flipud(noisy)
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            clean = np.rot90(clean, k)
            noisy = np.rot90(noisy, k)
        
        return clean, noisy
    
    
    
if __name__ == "__main__":
    print("=" * 70)
    print("Testing DenoisingDataset")
    print("=" * 70)
    
    # Create dataset with optimized settings
    dataset = DenoisingDataset(
        data_root='../data',
        split='train',
        categories=['xray'],  # ✅ Start with just one category for testing
        noise_levels=[25],     # ✅ Just one noise level for testing
        patch_size=40,
        stride=40,             # ✅ Non-overlapping patches
        augment=True,
        grayscale=True,
        max_patches_per_image=20,  # ✅ Only 20 patches per image
        max_images=100,  # ✅ ONLY 100 IMAGES FOR TESTING!
        lazy_load=True  # ✅ Load patches on-the-fly to save memory!
    )
    
    print(f"\nDataset size: {len(dataset)} patches")
    
    # Test getting a single patch
    noisy, clean = dataset[0]
    print(f"\nSample patch:")
    print(f"  Noisy shape: {noisy.shape}")
    print(f"  Clean shape: {clean.shape}")
    print(f"  Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    noisy_batch, clean_batch = next(iter(dataloader))
    
    print(f"\nBatch test:")
    print(f"  Noisy batch shape: {noisy_batch.shape}")
    print(f"  Clean batch shape: {clean_batch.shape}")
    
    print("\n✓ DenoisingDataset test passed!")