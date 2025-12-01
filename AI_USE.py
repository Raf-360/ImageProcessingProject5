# import torch
# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
# from tkinter import Label, Button
# from PIL import Image, ImageTk
# from pathlib import Path
# import segmentation_models_pytorch as smp
# from pathlib import Path

# # ==========================================================
# # Load model (MobileNetV2 UNet)
# # ==========================================================

# SCRIPT_DIR = Path(__file__).parent
# MODEL_PATH = SCRIPT_DIR / "xray_denoiser_mobilev2.pth"

# def load_model():
#     print("[INFO] Loading model...")
#     model = smp.Unet(
#         encoder_name="mobilenet_v2",
#         in_channels=1,
#         classes=1
#     )

#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#     print("[INFO] Model loaded.")
#     return model


# # ==========================================================
# # Preprocess image for the model
# # ==========================================================
# def preprocess(img):
#     img = cv2.resize(img, (512, 512))
#     img = img.astype("float32") / 255.0
#     tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#     return tensor


# # ==========================================================
# # Postprocess output
# # ==========================================================
# def postprocess(tensor):
#     img = tensor.squeeze().detach().numpy()
#     img = np.clip(img, 0, 1)
#     img = (img * 255).astype("uint8")
#     return img


# # ==========================================================
# # Noise Map (heatmap of noisy pixels)
# # ==========================================================
# def compute_noise_map(noisy, denoised):
#     diff = np.abs(noisy.astype("float32") - denoised.astype("float32"))
#     diff = diff / diff.max()
#     diff = (diff * 255).astype("uint8")
#     heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
#     return heat


# # ==========================================================
# # Convert an OpenCV image → Tkinter display format
# # ==========================================================
# def to_tk(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
#     img_pil = Image.fromarray(img_rgb)
#     return ImageTk.PhotoImage(img_pil)


# # ==========================================================
# # Inference & Display Logic
# # ==========================================================
# def run_denoise():
#     global model, panel_noisy, panel_noise, panel_clean

#     file_path = filedialog.askopenfilename(
#         title="Select Noisy Image",
#         filetypes=[("PNG Images", "*.png")]
#     )
#     if not file_path:
#         return

#     print("[INFO] Loading noisy image:", file_path)
#     noisy = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

#     # Preprocess
#     inp = preprocess(noisy)

#     # Inference
#     with torch.no_grad():
#         pred = model(inp)
#     denoised = postprocess(pred)

#     # Noise detection map
#     noise_map = compute_noise_map(
#         cv2.resize(noisy, (512,512)),
#         denoised
#     )

#     # Convert for GUI
#     noisy_display   = cv2.resize(noisy, (512,512))
#     noisy_tk        = to_tk(noisy_display)
#     noise_map_tk    = to_tk(noise_map)
#     denoised_tk     = to_tk(denoised)

#     # Update GUI
#     panel_noisy.config(image=noisy_tk)
#     panel_noisy.image = noisy_tk

#     panel_noise.config(image=noise_map_tk)
#     panel_noise.image = noise_map_tk

#     panel_clean.config(image=denoised_tk)
#     panel_clean.image = denoised_tk

#     print("[INFO] Denoising complete.")


# # ==========================================================
# # GUI Layout
# # ==========================================================
# model = load_model()

# root = tk.Tk()
# root.title("X-Ray Denoising AI")

# btn = Button(root, text="Select Noisy Image", command=run_denoise, font=("Arial", 14))
# btn.pack(pady=10)

# # Panels to display images
# panel_noisy = Label(root)
# panel_noisy.pack(side="left", padx=10, pady=10)

# panel_noise = Label(root)
# panel_noise.pack(side="left", padx=10, pady=10)

# panel_clean = Label(root)
# panel_clean.pack(side="left", padx=10, pady=10)

# root.mainloop()


import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# DnCNN MODEL (same as training)
# ============================================================
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()

        layers = []
        layers.append(nn.Conv2d(channels, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, 3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(model_path, device="cuda"):
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model: {model_path}")
    return model


# ============================================================
# DENOISE FUNCTION
# ============================================================
def denoise_image(model, img_path, output_path, device="cuda"):
    # Read image
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 640))

    original = img.copy()

    # Normalize
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        denoised_tensor = model(img_tensor)

    # Back to uint8
    denoised = denoised_tensor.squeeze().cpu().numpy()
    denoised = (denoised * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), denoised)

    print(f"[INFO] Saved denoised image to: {output_path}")
    return original, denoised


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    model_path = r"C:\\Users\\rafareye\\OneDrive - Texas Tech University\\Fall 2025\\Image Processing\\DnCNN_10_SaP.pth"

    # image to denoise
    img_path = r"C:\\Users\\rafareye\\Documents\\classes\\XRAY_Images\\S_A_P_10percent Output\\noisy_00000001_000.png"
    out_path = r"C:\\Users\\rafareye\\Documents\\classes\\XRAY_Images\\AI output\\Salt and Pepper Output DNCNN\\test_01_denoised.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device)
    original, denoised = denoise_image(model, img_path, out_path, device)
