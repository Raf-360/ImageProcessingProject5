import torch 
import torch.nn as nn

class DnCNNWithBias(nn.Module):
    """DnCNN with bias=True - for loading old checkpoints trained with bias."""
    
    def __init__(self, in_channels: int = 3, depth: int = 17, k_size: int = 3, n_filters: int = 64):
        """
        Initialize the Denoising CNN with bias enabled

        Args:
            in_channels (int): Number of input/output channels. Defaults to 3.
                1 -> grayscale
                3 -> RGB
            depth (int): Number of convolutional layers. Defaults to 17.
            k_size (int): Convolutional kernel size. Defaults to 3.
            n_filters (int): Number of filters in hidden layers. Defaults to 64.
        """
        
        super(DnCNNWithBias, self).__init__()
        
        self.depth = depth
        self.in_channels = in_channels
        
        # calculate padding for k_size and image size 
        padding = (k_size - 1) // 2
        
        # build out layers 
        layers = []
        
        
        # ----- First Layers -----
        # Input noisey image (1 or 3 channels)
        # outout 64 feature maps 
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=k_size,
            padding=padding,
            bias=True
        ))
        layers.append(nn.ReLU(inplace=True))
        
        # ----- Middle Layers -----
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=k_size,
                padding=padding,
                bias=True
            ))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))
            
        
        # last layer
        layers.append(nn.Conv2d(
            in_channels=n_filters,
            out_channels=in_channels,
            kernel_size=k_size,
            padding=padding,
            bias=True
        ))
        
        # Combine all layers 
        self.dncnn = nn.Sequential(*layers)
        
        self._initialize_weights()
        
        
        
    def _initialize_weights(self):
        """
        Initialize network weights using orthogonal initialization.
        This helps with training stability and faster convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Orthogonal initialization for convolutional layers
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: scale=1, shift=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x):
        """
        Forward pass with residual learning.
        
        Key Idea: Instead of predicting clean image directly,
        predict the NOISE, then subtract it from input.
        
        Args:
            x: Input noisy image tensor, shape (B, C, H, W)
               B = batch size
               C = channels (1 or 3)
               H = height
               W = width
        
        Returns:
            Denoised image tensor, same shape as input
        """
        
        # prediect noise residual
        noise = self.dncnn(x)
        
        # Subtract the predictec noise 
        out = x - noise
        
        return out
    
    
    
    def predict_noise(self, x):
        """
        Predict only the noise component.
        Useful for visualizing what the network learned.
        
        Args:
            x: Noisy image tensor
        
        Returns:
            Predicted noise residual
        """
        return self.dncnn(x)
    
    
    
if __name__ == "__main__":
    print("=" * 70)
    print("Testing DnCNN Architecture")
    print("=" * 70)
    
    # Test grayscale model
    print("\n--- Grayscale Model ---")
    model_gray = DnCNN(in_channels=1, depth=17, n_filters=64)
    total_params = sum(p.numel() for p in model_gray.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    test_input_gray = torch.randn(1, 1, 256, 256)
    print(f"Input shape:  {test_input_gray.shape}")
    output_gray = model_gray(test_input_gray)
    print(f"Output shape: {output_gray.shape}")
    print(f"Output range: [{output_gray.min():.3f}, {output_gray.max():.3f}]")
    
    # Test RGB model
    print("\n--- RGB Model ---")
    model_rgb = DnCNN(in_channels=3, depth=17, n_filters=64)
    total_params = sum(p.numel() for p in model_rgb.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    test_input_rgb = torch.randn(1, 3, 256, 256)
    print(f"Input shape:  {test_input_rgb.shape}")
    output_rgb = model_rgb(test_input_rgb)
    print(f"Output shape: {output_rgb.shape}")
    print(f"Output range: [{output_rgb.min():.3f}, {output_rgb.max():.3f}]")
    
    print("\n✓ DnCNN test passed!")