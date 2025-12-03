"""
Standardize folder naming conventions across the dataset.
Converts all variations to consistent format:
  - synthetic/, xray/, jellyfish/
  - clean/
  - gaussian_noise_15_sigma/, gaussian_noise_25_sigma/, gaussian_noise_55_sigma/
"""

import shutil
from pathlib import Path
from collections import defaultdict

def scan_current_structure(data_root: Path):
    """Scan and report current naming structure."""
    print("=" * 70)
    print("Current Dataset Structure Analysis")
    print("=" * 70)
    
    structure = defaultdict(list)
    
    for split in ['train', 'test', 'validation']:
        split_path = data_root / split
        if not split_path.exists():
            continue
            
        print(f"\n{split.upper()}/")
        
        for item in sorted(split_path.iterdir()):
            if item.is_dir():
                print(f"  └── {item.name}/")
                structure[split].append(item.name)
                
                # List subfolders
                for subfolder in sorted(item.iterdir()):
                    if subfolder.is_dir():
                        file_count = len(list(subfolder.glob('*.*')))
                        print(f"      └── {subfolder.name}/ ({file_count} files)")
    
    return structure

def get_standard_category_name(folder_name: str) -> str:
    """Convert various category folder names to standard format."""
    folder_lower = folder_name.lower()
    
    if 'synthetic' in folder_lower:
        return 'synthetic'
    elif 'xray' in folder_lower or 'x-ray' in folder_lower or 'x_ray' in folder_lower:
        return 'xray'
    elif 'jellyfish' in folder_lower or 'jelly' in folder_lower:
        return 'jellyfish'
    elif 'natural' in folder_lower:
        return 'jellyfish'  # Migrate natural to jellyfish
    else:
        return None

def get_standard_subfolder_name(folder_name: str) -> str:
    """Convert various subfolder names to standard format."""
    folder_lower = folder_name.lower()
    
    # Clean folders
    if folder_name == 'clean' or folder_lower == 'clean':
        return 'clean'
    elif 'clean' in folder_lower or folder_lower.endswith('_images'):
        return 'clean'
    
    # Noisy folders - extract sigma value
    if 'gaussian' in folder_lower or 'noise' in folder_lower:
        # Extract number (sigma value)
        import re
        numbers = re.findall(r'\d+', folder_name)
        if numbers:
            sigma = numbers[0]  # Take first number found
            return f'gaussian_noise_{sigma}_sigma'
    
    return None

def rename_folder(old_path: Path, new_name: str, dry_run: bool = False) -> bool:
    """Rename a folder safely."""
    new_path = old_path.parent / new_name
    
    if old_path.name == new_name:
        return False  # Already correct
    
    if new_path.exists():
        print(f"    ⚠️  WARNING: {new_path} already exists, skipping rename")
        return False
    
    if dry_run:
        print(f"    Would rename: {old_path.name} → {new_name}")
    else:
        old_path.rename(new_path)
        print(f"    ✓ Renamed: {old_path.name} → {new_name}")
    
    return True

def standardize_dataset(data_root: Path, dry_run: bool = False):
    """Standardize all folder names in the dataset."""
    print("\n" + "=" * 70)
    print("Dataset Standardization")
    print("=" * 70)
    print(f"Root: {data_root}")
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
    print("=" * 70)
    
    changes_made = 0
    
    for split in ['train', 'test', 'validation']:
        split_path = data_root / split
        if not split_path.exists():
            continue
        
        print(f"\n[{split.upper()}]")
        
        # Get all category folders
        category_folders = [f for f in split_path.iterdir() if f.is_dir()]
        
        for category_folder in sorted(category_folders):
            standard_category = get_standard_category_name(category_folder.name)
            
            if not standard_category:
                print(f"  ⚠️  Unknown category: {category_folder.name}, skipping")
                continue
            
            print(f"\n  Processing: {category_folder.name}/")
            
            # First, standardize subfolders
            subfolders = [f for f in category_folder.iterdir() if f.is_dir()]
            
            for subfolder in sorted(subfolders):
                standard_subfolder = get_standard_subfolder_name(subfolder.name)
                
                if standard_subfolder:
                    if rename_folder(subfolder, standard_subfolder, dry_run):
                        changes_made += 1
                else:
                    print(f"    ⚠️  Unknown subfolder pattern: {subfolder.name}")
            
            # Then, rename category folder if needed
            if rename_folder(category_folder, standard_category, dry_run):
                changes_made += 1
    
    print("\n" + "=" * 70)
    if dry_run:
        print(f"✓ Preview complete! Would make {changes_made} changes")
    else:
        print(f"✓ Standardization complete! Made {changes_made} changes")
    print("=" * 70)
    
    print("\nStandard structure:")
    print("data/")
    print("├── train/")
    print("│   ├── synthetic/")
    print("│   │   ├── clean/")
    print("│   │   ├── gaussian_noise_15_sigma/")
    print("│   │   ├── gaussian_noise_25_sigma/")
    print("│   │   └── gaussian_noise_55_sigma/")
    print("│   ├── xray/")
    print("│   │   └── [same structure]")
    print("│   └── jellyfish/")
    print("│       └── [same structure]")
    print("├── test/")
    print("│   └── [same structure]")
    print("└── validation/")
    print("    └── [same structure]")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Standardize dataset folder naming conventions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Standardizes to:
  Categories: synthetic/, xray/, jellyfish/
  Clean:      clean/
  Noisy:      gaussian_noise_{sigma}_sigma/

Examples:
  # Preview changes
  python standardize_names.py --dry-run
  
  # Apply standardization
  python standardize_names.py
  
  # Custom data root
  python standardize_names.py --data-root /path/to/data
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../../data",
        help="Root data folder (default: ../../data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan and report current structure"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"❌ ERROR: Data root '{data_root}' does not exist!")
        return
    
    # Always scan first
    scan_current_structure(data_root)
    
    if args.scan_only:
        print("\n✓ Scan complete (use without --scan-only to standardize)")
        return
    
    # Confirm before making changes
    if not args.dry_run:
        print("\n" + "=" * 70)
        response = input("Proceed with renaming? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    standardize_dataset(data_root, dry_run=args.dry_run)

if __name__ == "__main__":
    main()