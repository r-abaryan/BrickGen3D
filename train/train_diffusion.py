"""
Train Diffusion Model
=====================
Training script for the diffusion-based generator.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.text_encoder import TextEncoder
from src.diffusion_model import UNet3D, VoxelDiffusion
from src.dataset import SimpleShapeDataset


def train_diffusion(
    num_epochs=50,
    batch_size=64,  # Smaller batch for diffusion (more memory intensive)
    num_train_samples=5000,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Training Diffusion Model on {device}")
    
    # Initialize
    text_encoder = TextEncoder()
    model = UNet3D(text_dim=384, base_channels=32).to(device)
    diffusion = VoxelDiffusion(model, num_steps=50, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataset
    dataset = SimpleShapeDataset(num_samples=num_train_samples, voxel_size=32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")
        for texts, voxels in pbar:
            voxels = voxels.to(device)
            
            # Encode text
            with torch.no_grad():
                text_emb = text_encoder.encode(texts).to(device)
            
            # Random timestep
            t = torch.randint(0, diffusion.num_steps, (voxels.shape[0],), device=device)
            
            # Add noise
            noisy_voxels, noise = diffusion.add_noise(voxels, t)
            
            # Predict noise
            predicted_noise = model(noisy_voxels, t, text_emb)
            
            # Loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/diffusion_epoch_{epoch}.pth')
    
    # Save final
    torch.save(model.state_dict(), 'checkpoints/diffusion_final.pth')
    print("Training complete!")


if __name__ == "__main__":
    train_diffusion()

