"""
DnCNN Training Script for HPCC
Train a denoising CNN on noisy/clean image pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from deep.dncnn import DnCNN
from deep.dataset import DenoisingDataset
from utils.metrics import calculate_psnr, calculate_ssim


class DnCNNTrainer:
    """Trainer class for DnCNN model."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        
        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoints']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, datasets, and training components
        self.model = self.build_model()
        self.train_loader, self.val_loader = self.build_dataloaders()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # TensorBoard
        if self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # Training state
        self.start_epoch = 0
        self.best_psnr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        print("=" * 70)
        print("DnCNN Trainer Initialized")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print("=" * 70)
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self) -> torch.device:
        """Setup compute device (GPU, MPS, or CPU)."""
        if self.config['device']['cuda']:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using Apple M2 GPU (MPS)")
            else:
                device = torch.device('cpu')
                print("Using CPU (no GPU available)")
        else:
            device = torch.device('cpu')
            print("Using CPU (GPU disabled in config)")
        return device
    
    def build_model(self) -> nn.Module:
        """Build DnCNN model."""
        model_config = self.config['model']
        
        data_config = self.config["data"]
        in_channels = 1 if data_config["grayscale"] else 3
        model = DnCNN(
            in_channels=in_channels, 
            depth=model_config['depth'],
            k_size=model_config['k_size'],
            n_filters=model_config['n_filters']
        )
        model = model.to(self.device)
        
        # Multi-GPU support
        if len(self.config['device']['device_ids']) > 1:
            model = nn.DataParallel(model, device_ids=self.config['device']['device_ids'])
        
        return model
    
    def build_dataloaders(self) -> tuple:
        """Build training and validation dataloaders."""
        data_config = self.config['data']
        
        # Training dataset
        train_dataset = DenoisingDataset(
            data_root=data_config['data_root'],
            split='train',
            categories=data_config['categories'],
            noise_levels=data_config['noise_levels'],
            patch_size=data_config['patch_size'],
            stride=data_config['stride'],
            augment=data_config['augment'],
            grayscale=data_config['grayscale'],
            max_patches_per_image=data_config['max_patches_per_image'],
            max_images=data_config['max_images'],
            lazy_load=data_config['lazy_load']
        )
        
        # Validation dataset
        val_dataset = DenoisingDataset(
            data_root=data_config['data_root'],
            split='validation',
            categories=data_config['categories'],
            noise_levels=data_config['noise_levels'],
            patch_size=data_config['patch_size'],
            stride=data_config['stride'],
            augment=False,  # No augmentation for validation
            grayscale=data_config['grayscale'],
            max_patches_per_image=20,  # Fewer patches for validation
            max_images=100,  # Limit validation set size
            lazy_load=data_config['lazy_load']
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            prefetch_factor=data_config.get('prefetch_factor', 2)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        return train_loader, val_loader
    
    def build_criterion(self) -> nn.Module:
        """Build loss function."""
        loss_type = self.config['training']['loss']
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        opt_config = self.config['training']
        
        if opt_config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=0.9,
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
    
    def build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_type = self.config['training'].get('lr_scheduler', None)
        
        if scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # maximize PSNR
                factor=self.config['training']['lr_factor'],
                patience=self.config['training']['lr_patience']
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            denoised = self.model(noisy)
            
            # Compute loss
            loss = self.criterion(denoised, clean)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['clip_grad_norm']
                )
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config['logging']['log_every'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        for noisy, clean in tqdm(self.val_loader, desc='Validation'):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            denoised = self.model(noisy)
            
            # Compute loss
            loss = self.criterion(denoised, clean)
            total_loss += loss.item()
            
            # Compute metrics (convert to numpy for metric calculation)
            denoised_np = denoised.cpu().numpy()
            clean_np = clean.cpu().numpy()
            
            for i in range(denoised_np.shape[0]):
                # Convert from (C, H, W) to (H, W) or (H, W, C)
                if denoised_np.shape[1] == 1:
                    denoised_img = denoised_np[i, 0]
                    clean_img = clean_np[i, 0]
                else:
                    denoised_img = np.transpose(denoised_np[i], (1, 2, 0))
                    clean_img = np.transpose(clean_np[i], (1, 2, 0))
                
                # Convert to uint8 for metrics
                denoised_img = (np.clip(denoised_img, 0, 1) * 255).astype(np.uint8)
                clean_img = (np.clip(clean_img, 0, 1) * 255).astype(np.uint8)
                
                psnr = calculate_psnr(denoised_img, clean_img)
                ssim = calculate_ssim(denoised_img, clean_img)
                
                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        return {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save periodic checkpoint
        if epoch % self.config['checkpoints']['save_every'] == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (PSNR: {self.best_psnr:.2f} dB)")
    
    def train(self):
        """Main training loop."""
        print("\n🚀 Starting training...")
        print("=" * 70)
        
        for epoch in range(self.start_epoch + 1, self.config['training']['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['validation']['validate_every'] == 0:
                val_metrics = self.validate(epoch)
                
                # Log metrics
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                print(f"  Val Loss:   {val_metrics['loss']:.6f}")
                print(f"  Val PSNR:   {val_metrics['psnr']:.2f} dB")
                print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
                
                if self.writer:
                    self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
                    self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['psnr'])
                    else:
                        self.scheduler.step()
                
                # Check if best model
                is_best = val_metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics['psnr']
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Early stopping
                if self.config['training'].get('early_stopping'):
                    if self.patience_counter >= self.config['training']['patience']:
                        print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                        print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
                        break
        
        print("\n" + "=" * 70)
        print("✓ Training complete!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
        print("=" * 70)
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DnCNN on HPCC')
    parser.add_argument('--config', type=str, default='configs/dncnn_train.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train
    trainer = DnCNNTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
