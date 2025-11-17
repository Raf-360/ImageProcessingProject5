import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import filedialog

import cv2
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm


###############################################################
# 1. Dataset
###############################################################
class XrayDenoiseDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, limit=50000):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)

        # Limit number of images for speed
        self.clean_paths = sorted(self.clean_dir.glob("*.png"))[:limit]

        if len(self.clean_paths) == 0:
            raise RuntimeError(f"No PNG images found: {self.clean_dir}")

        print(f"[DEBUG] Found {len(self.clean_paths)} clean images")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        noisy_name = f"noisy_{clean_path.name}"
        mask_name  = f"noisy_{clean_path.stem}_mask.png"

        noisy_path = self.noisy_dir / noisy_name
        mask_path  = self.noisy_dir / mask_name

        # Load grayscale images
        clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if clean is None or noisy is None or mask is None:
            raise ValueError(f"[ERROR] Missing file near: {clean_path.name}")

        # Resize for speed
        clean = cv2.resize(clean, (512, 512)).astype("float32") / 255.0
        noisy = cv2.resize(noisy, (512, 512)).astype("float32") / 255.0
        mask  = cv2.resize(mask,  (512, 512)).astype("float32") / 255.0

        # Return as CHW tensors
        return (
            torch.tensor(noisy).unsqueeze(0),
            torch.tensor(clean).unsqueeze(0),
            torch.tensor(mask).unsqueeze(0),
        )


###############################################################
# 2. Masked Loss Function
###############################################################
def masked_l1(pred, target, mask):
    return (torch.abs(pred - target) * mask).mean()


###############################################################
# 3. Folder Picker
###############################################################
def pick_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    if not folder:
        raise RuntimeError(f"You must select a folder: {title}")
    return folder


###############################################################
# 4. Training Function
###############################################################
def train():
    print("\nSelect CLEAN XRAY_images folder:")
    clean_dir = pick_folder("Select XRAY_images folder")

    print("\nSelect NOISY + MASK folder:")
    noisy_dir = pick_folder("Select noisy folder")

    print("\n[DEBUG] clean_dir =", clean_dir)
    print("[DEBUG] noisy_dir =", noisy_dir)

    # Dataset
    dataset = XrayDenoiseDataset(clean_dir, noisy_dir, limit=15000)

    # DataLoader (Windows-safe)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,   # IMPORTANT for Windows
        pin_memory=True
    )

    # Fast encoder (MobileNetV2)
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        in_channels=1,
        classes=1
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print("\nStarting training...\n")

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0

        for noisy, clean, mask in tqdm(loader, desc=f"Epoch {epoch+1}/20"):
            noisy, clean, mask = noisy.cuda(), clean.cuda(), mask.cuda()

            with torch.cuda.amp.autocast():
                pred = model(noisy)
                loss = masked_l1(pred, clean, mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20 — Loss: {total_loss/len(loader):.6f}")

    # Save model
    save_path = Path(clean_dir).parent / "xray_denoiser_mobilev2.pth"
    torch.save(model.state_dict(), save_path)

    print("\nModel saved to:", save_path)


###############################################################
# MAIN ENTRY
###############################################################
if __name__ == "__main__":
    train()
