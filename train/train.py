"""
Training Script for BrickGen3D
===============================
This script trains the text-to-3D voxel generator.

Training Process:
1. Load text encoder (pre-trained, frozen)
2. Create voxel generator (to be trained)
3. Create dataset of shapes with descriptions
4. Train using Binary Cross-Entropy loss
5. Save checkpoints

Key Parameters:
- Epochs: 50 (quick training, ~10-15 min on CPU)
- Batch size: 16 (small for CPU/small GPU)
- Learning rate: 0.001
- Dataset: 5000 synthetic shapes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from src.text_encoder import TextEncoder
from src.voxel_generator import VoxelGenerator, count_parameters
from src.dataset import SimpleShapeDataset
from src.voxel_utils import visualize_voxels, get_voxel_stats


class BrickGen3DTrainer:
    """
    Handles the training process.
    
    Loss Function:
    We use Binary Cross-Entropy (BCE) because each voxel is binary (occupied/empty).
    BCE = -[y*log(p) + (1-y)*log(1-p)]
    where y is ground truth (0 or 1) and p is predicted probability.
    """
    
    def __init__(
        self,
        text_dim=384,
        voxel_size=32,
        hidden_dim=512,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize text encoder (pre-trained, frozen)
        print("\n=== Initializing Text Encoder ===")
        self.text_encoder = TextEncoder()
        
        # Initialize voxel generator (to be trained)
        print("\n=== Initializing Voxel Generator ===")
        self.model = VoxelGenerator(
            text_dim=text_dim,
            voxel_size=voxel_size,
            hidden_dim=hidden_dim
        ).to(device)
        
        num_params = count_parameters(self.model)
        print(f"Model parameters: {num_params:,}")
        print(f"Model size: ~{num_params * 4 / 1024 / 1024:.1f} MB")
        
        # Loss function
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy
        
        # Optimizer (Adam is a good default)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,  # Learning rate
            betas=(0.9, 0.999),
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler (reduce LR when loss plateaus)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (texts, voxels) in enumerate(pbar):
            # Move voxels to device
            voxels = voxels.to(self.device)
            
            # Encode text (no gradients needed for encoder)
            with torch.no_grad():
                text_embeddings = self.text_encoder.encode(texts).to(self.device)
            
            # Forward pass
            predicted_voxels = self.model(text_embeddings)
            
            # Calculate loss
            loss = self.criterion(predicted_voxels, voxels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for texts, voxels in dataloader:
                voxels = voxels.to(self.device)
                text_embeddings = self.text_encoder.encode(texts).to(self.device)
                
                predicted_voxels = self.model(text_embeddings)
                loss = self.criterion(predicted_voxels, voxels)
                
                val_loss += loss.item()
        
        avg_loss = val_loss / len(dataloader)
        return avg_loss
    
    def train(
        self,
        num_epochs=50,
        batch_size=128,
        num_train_samples=5000,
        num_val_samples=500,
        save_dir='checkpoints'
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_train_samples: Number of training samples
            num_val_samples: Number of validation samples
            save_dir: Directory to save checkpoints
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Create datasets
        print("\nCreating datasets...")
        train_dataset = SimpleShapeDataset(num_samples=num_train_samples, voxel_size=32)
        val_dataset = SimpleShapeDataset(num_samples=num_val_samples, voxel_size=32)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4  # Use 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {len(train_loader)}")
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print statistics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = save_dir / 'best_model.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
        
        # Save final model
        final_path = save_dir / 'final_model.pth'
        self.save_checkpoint(final_path, num_epochs, val_loss)
        print(f"\n✓ Training complete! Final model saved to {final_path}")
        
        # Plot training curves
        self.plot_training_curves(save_dir / 'training_curves.png')
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_path):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")


def main():
    """Main training function."""
    print("="*60)
    print("BrickGen3D Training")
    print("="*60)
    
    # Initialize trainer
    trainer = BrickGen3DTrainer(
        text_dim=384,
        voxel_size=32,
        hidden_dim=512
    )
    
    # Train
    trainer.train(
        num_epochs=50,          # Number of epochs (increase for better results)
        batch_size=128,          # Batch size (increase if you have GPU)
        num_train_samples=5000, # Training samples
        num_val_samples=500,    # Validation samples
        save_dir='checkpoints'
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check 'checkpoints/best_model.pth' for the trained model")
    print("2. Run 'python generate.py' to generate 3D models from text")
    print("3. View 'checkpoints/training_curves.png' to see training progress")


if __name__ == "__main__":
    main()

