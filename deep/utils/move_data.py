"""
Reorganize data into train/test/validation splits (80/10/10).
Maintains original data distribution (XRAY-focused).
"""

import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict

def get_image_files(directory: Path):
    """Get all image files from a directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    if not directory.exists():
        return []
    return [f for f in directory.iterdir() if f.suffix.lower() in extensions and f.is_file()]

def categorize_images(train_path: Path):
    """
    Categorize images by type (synthetic, xray, jellyfish) and noise level.
    
    Returns:
        Dict[category][noise_level] = list of image paths
    """
    categories = defaultdict(lambda: defaultdict(list))
    
    # Map folders to categories
    folder_mapping = {
        'synthetic': 'synthetic',
        'xray': 'xray',
        'jellyfish': 'jellyfish'
    }
    
    print("\nScanning directories:")
    
    # Scan train folder for all image categories
    for category_folder in train_path.iterdir():
        if not category_folder.is_dir():
            continue
            
        category = folder_mapping.get(category_folder.name, None)
        if not category:
            print(f"  Skipping unknown folder: {category_folder.name}")
            continue
        
        print(f"\n  Checking {category_folder.name}/")
        
        # Get clean images - check both 'clean' folder and category-specific folders
        # (e.g., 'XRAY_images' for xray, 'clean_images' for synthetic)
        clean_candidates = [
            category_folder / 'clean',
            category_folder / f'{category}_images',
            category_folder / f'{category.upper()}_images',
            category_folder / 'clean_images'
        ]
        
        for clean_folder in clean_candidates:
            if clean_folder.exists():
                images = get_image_files(clean_folder)
                if images:
                    categories[category]['clean'] = images
                    print(f"    ✓ {len(images)} clean images from {clean_folder.name}/")
                    break
        
        # Get noisy images at different sigma levels
        # Handle both 'gaussian_noise_*' and 'CATEGORY_gaussian_noise_*' patterns
        noise_patterns = [
            'gaussian_noise_*',
            f'{category}_gaussian_noise_*',
            f'{category.upper()}_gaussian_noise_*'
        ]
        
        found_folders = set()
        for pattern in noise_patterns:
            for noisy_folder in category_folder.glob(pattern):
                if noisy_folder in found_folders:
                    continue
                found_folders.add(noisy_folder)
                
                # Extract sigma value from folder name
                folder_name = noisy_folder.name
                # Handle both 'gaussian_noise_25_sigma' and 'XRAY_gaussian_noise_25_sigma'
                parts = folder_name.split('_')
                sigma_idx = parts.index('noise') + 1 if 'noise' in parts else -1
                if sigma_idx > 0 and sigma_idx < len(parts):
                    sigma = parts[sigma_idx]
                    
                    images = get_image_files(noisy_folder)
                    if images:
                        categories[category][f'noisy_{sigma}'] = images
                        print(f"    ✓ {len(images)} noisy (σ={sigma}) images from {noisy_folder.name}/")
    
    return categories

def split_data(images, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Split images into train/test/validation sets.
    
    Args:
        images: List of image paths
        train_ratio: Proportion for training (default: 0.8)
        test_ratio: Proportion for testing (default: 0.1)
        val_ratio: Proportion for validation (default: 0.1)
    
    Returns:
        Tuple of (train_list, test_list, val_list)
    """
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n = len(images)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    
    train_idx = indices[:train_end]
    test_idx = indices[train_end:test_end]
    val_idx = indices[test_end:]
    
    return ([images[i] for i in train_idx],
            [images[i] for i in test_idx],
            [images[i] for i in val_idx])

def copy_files(file_list, dest_folder: Path):
    """Copy files to destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    skipped = 0
    for src_file in file_list:
        dest_file = dest_folder / src_file.name
        
        # Skip if source and destination are the same
        try:
            if src_file.resolve() == dest_file.resolve():
                skipped += 1
                continue
        except (OSError, ValueError):
            pass  # Handle cases where resolve fails
        
        # Skip if destination exists and is identical
        if dest_file.exists():
            if dest_file.stat().st_size == src_file.stat().st_size:
                skipped += 1
                continue
        
        shutil.copy2(src_file, dest_file)
        copied += 1
    
    if skipped > 0:
        print(f"    (skipped {skipped} existing files)")
    
    return copied

def reorganize_dataset(data_root: Path, dry_run=False):
    """
    Reorganize dataset from data/train/ into train/test/validation splits.
    
    Args:
        data_root: Root data folder containing train/, test/, validation/
        dry_run: If True, only preview without copying
    """
    print("=" * 70)
    print("Dataset Reorganization Script")
    print("=" * 70)
    print(f"Root folder: {data_root}")
    print(f"Source: {data_root}/train/")
    print(f"Destinations: {data_root}/{{train,test,validation}}/")
    print(f"Split: 80% train, 10% test, 10% validation")
    print(f"Mode: Keep original data distribution (XRAY-focused)")
    if dry_run:
        print("🔍 DRY RUN - No files will be copied")
    print("=" * 70)
    
    train_path = data_root / 'train'
    
    if not train_path.exists():
        print(f"\n❌ ERROR: {train_path} does not exist!")
        return
    
    # Step 1: Categorize existing images in train/
    print("\n[1/3] Categorizing images in train/...")
    categories = categorize_images(train_path)
    
    if not categories:
        print("\n❌ ERROR: No images found in train folder!")
        return
    
    # Step 2: Split data
    print("\n[2/3] Splitting data (80/10/10)...")
    np.random.seed(42)  # For reproducibility
    
    splits = {
        'train': defaultdict(lambda: defaultdict(list)),
        'test': defaultdict(lambda: defaultdict(list)),
        'validation': defaultdict(lambda: defaultdict(list))
    }
    
    total_stats = {'train': 0, 'test': 0, 'validation': 0}
    category_stats = defaultdict(lambda: {'train': 0, 'test': 0, 'validation': 0})
    
    for category, noise_levels in categories.items():
        clean_images = noise_levels.get('clean', [])
        
        if not clean_images:
            print(f"\n⚠️  WARNING: No clean images for {category}, skipping...")
            continue
        
        # Split clean images first to get the image names
        train_clean, test_clean, val_clean = split_data(clean_images, 0.8, 0.1, 0.1)
        train_names = {img.name for img in train_clean}
        test_names = {img.name for img in test_clean}
        val_names = {img.name for img in val_clean}
        
        print(f"\n{category.upper()}:")
        print(f"  Train: {len(train_clean):4d} | Test: {len(test_clean):4d} | Val: {len(val_clean):4d}")
        
        category_stats[category]['train'] = len(train_clean)
        category_stats[category]['test'] = len(test_clean)
        category_stats[category]['validation'] = len(val_clean)
        
        total_stats['train'] += len(train_clean)
        total_stats['test'] += len(test_clean)
        total_stats['validation'] += len(val_clean)
        
        # Split all noise levels based on clean image names
        for noise_level, images in noise_levels.items():
            splits['train'][category][noise_level] = [
                img for img in images if img.name in train_names
            ]
            splits['test'][category][noise_level] = [
                img for img in images if img.name in test_names
            ]
            splits['validation'][category][noise_level] = [
                img for img in images if img.name in val_names
            ]
    
    # Step 3: Copy files to new structure
    print("\n[3/3] Copying files...")
    
    if dry_run:
        print("\n🔍 DRY RUN - Showing what would be copied:\n")
    
    total_copied = 0
    for split_name, category_data in splits.items():
        split_folder = data_root / split_name
        
        for category, noise_levels in category_data.items():
            # Map back to original folder names
            folder_name = {
                'synthetic': 'synthetic',
                'xray': 'xray',
                'jellyfish': 'jellyfish'
            }[category]
            
            for noise_level, images in noise_levels.items():
                if images:
                    dest_folder = split_folder / folder_name / noise_level
                    
                    if dry_run:
                        print(f"  Would copy {len(images):4d} to {split_name}/{folder_name}/{noise_level}/")
                    else:
                        copied = copy_files(images, dest_folder)
                        total_copied += copied
                        print(f"  Copied {copied:4d} to {split_name}/{folder_name}/{noise_level}/")
    
    print("\n" + "=" * 70)
    if dry_run:
        print(f"✓ Preview complete! Would copy {total_copied} files")
    else:
        print(f"✓ Dataset reorganization complete! ({total_copied} files copied)")
    print("=" * 70)
    
    # Print summary
    print("\nDataset Summary (clean images):")
    total = sum(total_stats.values())
    print(f"  Train:      {total_stats['train']:4d} images ({total_stats['train']*100/total:.1f}%)")
    print(f"  Test:       {total_stats['test']:4d} images ({total_stats['test']*100/total:.1f}%)")
    print(f"  Validation: {total_stats['validation']:4d} images ({total_stats['validation']*100/total:.1f}%)")
    print(f"  Total:      {total:4d} images")
    
    print("\nPer-category distribution:")
    for category in ['synthetic', 'xray', 'jellyfish']:
        if category in category_stats:
            stats = category_stats[category]
            cat_total = sum(stats.values())
            print(f"  {category.upper()}:")
            print(f"    Train: {stats['train']:4d} | Test: {stats['test']:4d} | Val: {stats['validation']:4d} | Total: {cat_total:4d}")
    
    print("\nFinal structure:")
    print(f"{data_root}/")
    print("├── train/ (80%)")
    print("│   ├── synthetic/")
    print("│   │   ├── clean/")
    print("│   │   ├── gaussian_noise_15_sigma/")
    print("│   │   ├── gaussian_noise_25_sigma/")
    print("│   │   └── gaussian_noise_50_sigma/")
    print("│   ├── xray/ (largest)")
    print("│   │   └── [same structure]")
    print("│   └── jellyfish/")
    print("│       └── [same structure]")
    print("├── test/ (10%)")
    print("│   └── [same structure]")
    print("└── validation/ (10%)")
    print("    └── [same structure]")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize dataset: split data/train/ into train/test/validation (80/10/10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview the split without copying
  python reorganize_data.py --dry-run
  
  # Run the reorganization
  python reorganize_data.py
  
  # Specify custom data root
  python reorganize_data.py --data-root /path/to/data
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root data folder containing train/ (default: data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the split without copying files"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"❌ ERROR: Data root '{data_root}' does not exist!")
        return
    
    if not args.dry_run:
        # Check if test/validation folders already have data
        test_path = data_root / 'test'
        val_path = data_root / 'validation'
        
        has_data = False
        for folder in [test_path, val_path]:
            if folder.exists() and any(folder.rglob('*.png')) or any(folder.rglob('*.jpg')):
                has_data = True
                break
        
        if has_data:
            print("\n⚠️  WARNING: test/ or validation/ folders already contain images!")
            response = input("This will add/overwrite data. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    reorganize_dataset(data_root, dry_run=args.dry_run)

if __name__ == "__main__":
    main()