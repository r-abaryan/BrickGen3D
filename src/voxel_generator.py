"""
Voxel Generator Model
=====================
This is the core model that generates 3D voxel grids from text embeddings.

Architecture:
    Text Embedding (384) → MLP → Voxel Grid (32x32x32)
    
We use a simple but effective architecture:
1. Multiple fully connected layers to expand the embedding
2. Reshape to 3D grid
3. Refinement with 3D convolutions
4. Sigmoid activation for occupancy probability

This is MUCH simpler than diffusion models or GANs, making it:
- Easy to understand
- Fast to train
- Lightweight (only ~10M parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelGenerator(nn.Module):
    """
    Generates 3D voxel grids from text embeddings.
    
    Voxel Grid: Each cell is either occupied (1) or empty (0)
    Output: 32x32x32 grid = 32,768 voxels (good balance of detail vs. speed)
    """
    
    def __init__(self, text_dim=384, voxel_size=32, hidden_dim=512):
        """
        Initialize the generator.
        
        Args:
            text_dim: Dimension of text embeddings (384 for MiniLM)
            voxel_size: Size of output voxel grid (32 means 32x32x32)
            hidden_dim: Size of hidden layers
        """
        super(VoxelGenerator, self).__init__()
        
        self.voxel_size = voxel_size
        self.text_dim = text_dim
        
        # === PART 1: MLP to expand text embedding ===
        # We gradually expand from 384 → 512 → 1024 → 4096
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, 4 * 4 * 4 * 64)  # 4096 values
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        # === PART 2: 3D Convolutions for refinement ===
        # Start with 4x4x4 grid with 64 channels
        # Upsample to 32x32x32 with 1 channel
        
        # 4x4x4 → 8x8x8
        self.conv1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        
        # 8x8x8 → 16x16x16
        self.conv2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn3d2 = nn.BatchNorm3d(16)
        
        # 16x16x16 → 32x32x32
        self.conv3 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1)
        self.bn3d3 = nn.BatchNorm3d(8)
        
        # Final layer: reduce channels to 1
        self.conv4 = nn.Conv3d(8, 1, kernel_size=3, padding=1)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, text_embedding):
        """
        Generate voxel grid from text embedding.
        
        Args:
            text_embedding: Tensor of shape (batch_size, text_dim)
            
        Returns:
            voxel_grid: Tensor of shape (batch_size, 1, voxel_size, voxel_size, voxel_size)
                       Values are between 0 and 1 (occupancy probability)
        """
        batch_size = text_embedding.shape[0]
        
        # === PART 1: Expand embedding with MLP ===
        x = F.relu(self.bn1(self.fc1(text_embedding)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # Reshape to 3D: (batch, 4096) → (batch, 64, 4, 4, 4)
        x = x.view(batch_size, 64, 4, 4, 4)
        
        # === PART 2: Upsample with 3D convolutions ===
        # 4x4x4 → 8x8x8
        x = F.relu(self.bn3d1(self.conv1(x)))
        
        # 8x8x8 → 16x16x16
        x = F.relu(self.bn3d2(self.conv2(x)))
        
        # 16x16x16 → 32x32x32
        x = F.relu(self.bn3d3(self.conv3(x)))
        
        # Final convolution to get occupancy probability
        x = torch.sigmoid(self.conv4(x))  # Sigmoid: 0 (empty) to 1 (occupied)
        
        return x
    
    def generate(self, text_embedding, threshold=0.5):
        """
        Generate binary voxel grid (for visualization/export).
        
        Args:
            text_embedding: Text embedding tensor
            threshold: Occupancy threshold (0.5 means >50% probability = occupied)
            
        Returns:
            Binary voxel grid (0 or 1)
        """
        with torch.no_grad():
            voxel_prob = self.forward(text_embedding)
            voxel_binary = (voxel_prob > threshold).float()
        return voxel_binary


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    print("Testing VoxelGenerator...")
    
    # Create model
    model = VoxelGenerator(text_dim=384, voxel_size=32, hidden_dim=512)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # Test with random embedding
    batch_size = 2
    dummy_embedding = torch.randn(batch_size, 384)
    
    # Generate voxels
    output = model(dummy_embedding)
    print(f"\nInput shape: {dummy_embedding.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test binary generation
    binary_output = model.generate(dummy_embedding, threshold=0.5)
    print(f"Binary output shape: {binary_output.shape}")
    print(f"Occupied voxels: {binary_output.sum().item()} / {32*32*32}")

