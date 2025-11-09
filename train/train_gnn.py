"""
Train GNN Model
===============
Training script for GNN-based mesh generator.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh

from src.text_encoder import TextEncoder
from src.gnn_model import MeshGNN
from src.dataset import SimpleShapeDataset


def voxel_to_point_cloud(voxel_grid, num_points=162):
    """Convert voxel to point cloud for GNN target."""
    coords = torch.nonzero(voxel_grid.squeeze() > 0.5, as_tuple=False).float()
    
    if len(coords) == 0:
        return torch.zeros(num_points, 3)
    
    # Normalize to [-1, 1]
    coords = (coords / 16.0) - 1.0
    
    # Sample points if too many
    if len(coords) > num_points:
        indices = torch.randperm(len(coords))[:num_points]
        coords = coords[indices]
    
    # Pad if too few
    if len(coords) < num_points:
        padding = torch.zeros(num_points - len(coords), 3)
        coords = torch.cat([coords, padding], dim=0)
    
    return coords


def train_gnn(
    num_epochs=100,
    batch_size=64,
    num_train_samples=5000,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Training GNN Model on {device}")
    
    # Initialize
    text_encoder = TextEncoder()
    model = MeshGNN(text_dim=384, hidden_dim=256, num_layers=4).to(device)
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
            # Convert voxels to point clouds (target)
            targets = []
            for voxel in voxels:
                pc = voxel_to_point_cloud(voxel, num_points=model.template_vertices.shape[0])
                targets.append(pc)
            targets = torch.stack(targets).to(device)
            
            # Encode text
            with torch.no_grad():
                text_emb = text_encoder.encode(texts).to(device)
            
            # Forward
            predicted_vertices = model(text_emb)
            
            # Loss: Chamfer distance (simplified)
            loss = torch.nn.functional.mse_loss(predicted_vertices, targets)
            
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
            }, f'checkpoints/gnn_epoch_{epoch}.pth')
    
    # Save final
    torch.save(model.state_dict(), 'checkpoints/gnn_final.pth')
    print("Training complete!")


if __name__ == "__main__":
    train_gnn()

