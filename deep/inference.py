"""
DnCNN inference wrapper for main.py integration.
Compatible with existing CLI interface.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from deep.dncnn import DnCNN


class DnCNNDenoiser:
    """Wrapper class for DnCNN inference."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize DnCNN denoiser from checkpoint.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"DnCNN loaded from {checkpoint_path}")
        print(f"Running on: {self.device}")
    
    def _load_model(self) -> DnCNN:
        """Load model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model config from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            data_config = checkpoint['config']['data']
            
            # Determine channels from data config
            in_channels = 1 if data_config.get('grayscale', True) else 3
            
            model = DnCNN(
                in_channels=in_channels,
                depth=model_config['depth'],
                k_size=model_config['k_size'],
                n_filters=model_config['n_filters']
            )
        else:
            # Default architecture if config not in checkpoint
            print("Warning: Model config not found in checkpoint, using default architecture")
            model = DnCNN(in_channels=1, depth=17, k_size=3, n_filters=64)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrappers
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Load with strict=False to ignore bias terms from old models
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            noisy_image: Noisy image as numpy array (H, W) or (H, W, C)
                        Values should be in [0, 255] uint8 or [0, 1] float
        
        Returns:
            Denoised image with same shape and dtype as input
        """
        # Store original properties
        original_shape = noisy_image.shape
        original_dtype = noisy_image.dtype
        
        # Normalize to [0, 1] if needed
        if original_dtype == np.uint8:
            image = noisy_image.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            image = noisy_image.astype(np.float32)
            was_uint8 = False
        
        # Determine expected channels from model
        model_channels = self.model.in_channels
        
        # Handle channel conversion based on model expectations
        if len(image.shape) == 2:
            # Grayscale input (H, W)
            if model_channels == 3:
                # Model expects RGB, replicate grayscale to 3 channels
                image = np.stack([image] * 3, axis=2)
            # else: grayscale model, keep as is
        elif len(image.shape) == 3:
            # Color input (H, W, C)
            if image.shape[2] == 3 and model_channels == 1:
                # RGB input but model expects grayscale
                image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            elif image.shape[2] == 1 and model_channels == 3:
                # Single channel input but model expects RGB
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 1:
                # Squeeze single channel
                image = image[:, :, 0]
        
        # Convert to tensor with proper dimensions
        if len(image.shape) == 2:
            # Grayscale: (H, W) -> (1, 1, H, W)
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            # RGB: (H, W, 3) -> (1, 3, H, W)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        image_tensor = image_tensor.to(self.device)
        
        # Denoise
        denoised_tensor = self.model(image_tensor)
        
        # Convert back to numpy
        if model_channels == 1:
            # Grayscale: (1, 1, H, W) -> (H, W)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        else:
            # RGB: (1, 3, H, W) -> (H, W, 3)
            denoised = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clip to valid range
        denoised = np.clip(denoised, 0, 1)
        
        # Match original shape format
        if len(original_shape) == 3 and len(denoised.shape) == 2:
            # Original was (H, W, C) but result is (H, W), expand
            if original_shape[2] == 3:
                denoised = np.stack([denoised] * 3, axis=2)
            else:
                denoised = denoised[:, :, np.newaxis]
        elif len(original_shape) == 2 and len(denoised.shape) == 3:
            # Original was (H, W) but result is (H, W, C), convert to grayscale
            if denoised.shape[2] == 3:
                denoised = 0.299 * denoised[:, :, 0] + 0.587 * denoised[:, :, 1] + 0.114 * denoised[:, :, 2]
            else:
                denoised = denoised[:, :, 0]
        
        # Convert back to uint8 if needed
        if was_uint8:
            denoised = (denoised * 255).astype(np.uint8)
        else:
            denoised = denoised.astype(original_dtype)
        
        return denoised


# Global denoiser instance (lazy initialization)
_denoiser_instance = None


def dncnn_denoise(noisy_image: np.ndarray, 
                  checkpoint_path: Optional[str] = None,
                  **kwargs) -> np.ndarray:
    """
    Denoise image using DnCNN (compatible with main.py CLI).
    
    Args:
        noisy_image: Noisy image as numpy array
        checkpoint_path: Path to trained model checkpoint. 
                        If None, uses 'checkpoints/checkpoint_best.pth'
        **kwargs: Additional arguments (for compatibility, ignored)
    
    Returns:
        Denoised image as numpy array
    """
    global _denoiser_instance
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = 'checkpoints/checkpoint_best.pth'
    
    # Lazy initialization of denoiser
    if _denoiser_instance is None:
        _denoiser_instance = DnCNNDenoiser(checkpoint_path)
    
    return _denoiser_instance.denoise(noisy_image)


# For compatibility with existing methods
def apply_dncnn(image: np.ndarray, **kwargs) -> np.ndarray:
    """Alias for dncnn_denoise."""
    return dncnn_denoise(image, **kwargs)
