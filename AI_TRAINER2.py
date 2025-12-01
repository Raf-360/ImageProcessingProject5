import os
from pathlib import Path
import numpy as np
import cv2
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ============================================================
# 1. DnCNN MODEL DEFINITION
# ============================================================
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Final layer
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # DnCNN predicts noise → clean = noisy - noise
        noise = self.dncnn(x)
        return x - noise


# ============================================================
# 2. DATASET CLASS (Clean + Noisy)
# ============================================================
class SaPDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, target_size=640):
        self.clean_paths = sorted(glob(str(Path(clean_dir) / "*.png")))
        self.noisy_paths = sorted(glob(str(Path(noisy_dir) / "*.png")))
        self.target = target_size

        assert len(self.clean_paths) == len(self.noisy_paths)

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = cv2.imread(self.clean_paths[idx], cv2.IMREAD_GRAYSCALE)
        noisy = cv2.imread(self.noisy_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Resize to 640x640
        clean = cv2.resize(clean, (self.target, self.target), interpolation=cv2.INTER_AREA)
        noisy = cv2.resize(noisy, (self.target, self.target), interpolation=cv2.INTER_AREA)

        clean = clean.astype(np.float32) / 255.0
        noisy = noisy.astype(np.float32) / 255.0

        clean = torch.from_numpy(clean).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return noisy, clean



# ============================================================
# 3. TRAINING FUNCTION
# ============================================================
def train_dncnn(clean_dir, noisy_dir, epochs=20, batch_size=8, lr=1e-3):

    print("[INFO] Loading dataset...")
    dataset = SaPDataset(clean_dir, noisy_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0

        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss = {running_loss/len(loader):.6f}")

    # Save model
    save_path = "DnCNN_10_SaP.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")

    return model


# ============================================================
# 4. MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    clean = r"C:\\Users\\rafareye\\Documents\\classes\\XRAY_Images\\First_2k_XRAY_IMAGES"
    noisy = r"C:\\Users\\rafareye\\Documents\\classes\\XRAY_Images\\S_A_P_10percent Output"

    train_dncnn(clean, noisy, epochs=20, batch_size=4, lr=1e-3)
