import torch 
import torch.nn as nn

class DnCNN(nn.Module):
    
    def __init__(self, depth: int=17, k_size: int=5, n_channels: int=1, n_filters: int=64):
        """
        Initialize the Denoising CNN

        Args:
            depth (int, optional): Number of layers in the NN. Defaults to 17.
            k_size (int, optional): Convolutional Kernal Size. Defaults to 5.
            n_channels (int, optional):number of channels to process. Defaults to 1.
                1 -> grayscale
                3 -> RGB
        """
        
        super(DnCNN, self).__init__()
        
        self.depth = depth
        self.n_channels = n_channels
        
        # calculate padding for k_size and image size 
        padding = (k_size - 1) // 2
        
        # build out layers 
        layers = []
        
        
        # ----- First Layers -----
        # Input noisey image (1 or 3 channels)
        # outout 64 feature maps 
        layers.append(nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_filters,
            kernel_size=k_size,
            padding=padding,
            bias=False
        ))
        layers.append(nn.ReLU(inplace=True))
        
        # ----- Middle Layers -----
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=k_size,
                padding=padding,
                bias=False
            ))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))
            
        
        # last layer
        layers.append(nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_channels,
            kernel_size=k_size,
            padding=padding,
            bias=False
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
    
    # Create grayscale model
    model = DnCNN(depth=17, n_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Test with dummy data
    test_input = torch.randn(1, 1, 256, 256)  # 1 grayscale 256×256 image
    print(f"\nInput shape:  {test_input.shape}")
    
    # Forward pass
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✓ DnCNN test passed!")