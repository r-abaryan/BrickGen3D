"""
Diffusion Model for Text-to-3D
===============================
Uses iterative denoising for higher quality generation.

Architecture: UNet3D with text conditioning
Training: Add noise -> Predict noise -> Remove noise
Inference: Start from noise -> Denoise iteratively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep encoding for diffusion."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock3D(nn.Module):
    """3D Residual block with time and text conditioning."""
    def __init__(self, in_channels, out_channels, time_dim, text_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.text_mlp = nn.Linear(text_dim, out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, time_emb, text_emb):
        h = self.conv1(F.relu(x))
        
        # Add time conditioning
        time_emb = self.time_mlp(time_emb)[:, :, None, None, None]
        h = h + time_emb
        
        # Add text conditioning (scale by 1.5 for stronger influence)
        text_emb = self.text_mlp(text_emb)[:, :, None, None, None]
        h = h + text_emb * 1.5
        
        h = self.norm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.relu(h + self.shortcut(x))


class UNet3D(nn.Module):
    """3D UNet for voxel diffusion."""
    def __init__(self, text_dim=384, base_channels=32):
        super().__init__()
        
        self.time_dim = 128
        self.time_embed = SinusoidalPositionEmbeddings(self.time_dim)
        
        # Encoder
        self.enc1 = ResidualBlock3D(1, base_channels, self.time_dim, text_dim)
        self.enc2 = ResidualBlock3D(base_channels, base_channels*2, self.time_dim, text_dim)
        self.enc3 = ResidualBlock3D(base_channels*2, base_channels*4, self.time_dim, text_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_channels*4, base_channels*4, self.time_dim, text_dim)
        
        # Decoder
        self.dec3 = ResidualBlock3D(base_channels*8, base_channels*2, self.time_dim, text_dim)
        self.dec2 = ResidualBlock3D(base_channels*4, base_channels, self.time_dim, text_dim)
        self.dec1 = ResidualBlock3D(base_channels*2, base_channels, self.time_dim, text_dim)
        
        # Output
        self.out = nn.Conv3d(base_channels, 1, 1)
        
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, timestep, text_emb):
        # Time embedding
        t = self.time_embed(timestep)
        
        # Encoder
        e1 = self.enc1(x, t, text_emb)
        e2 = self.enc2(self.pool(e1), t, text_emb)
        e3 = self.enc3(self.pool(e2), t, text_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3), t, text_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1), t, text_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), t, text_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), t, text_emb)
        
        return self.out(d1)


class VoxelDiffusion:
    """Diffusion process wrapper."""
    def __init__(self, model, num_steps=50, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        """Add noise to voxels at timestep t."""
        noise = torch.randn_like(x)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise, noise
    
    @torch.no_grad()
    def sample(self, text_emb, shape=(1, 1, 32, 32, 32)):
        """Generate voxels by denoising."""
        # Start from pure noise
        x = torch.randn(shape).to(self.device)
        
        # Ensure text embedding is on correct device and shape
        if text_emb.shape[0] != shape[0]:
            text_emb = text_emb.expand(shape[0], -1)
        text_emb = text_emb.to(self.device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t_batch, text_emb)
            
            # Remove predicted noise
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            x = (x - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
            
            # Add noise (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise
        
        return torch.sigmoid(x)


# Example usage
if __name__ == "__main__":
    model = UNet3D(text_dim=384, base_channels=32)
    diffusion = VoxelDiffusion(model, num_steps=50)
    
    # Test
    text_emb = torch.randn(1, 384)
    voxels = diffusion.sample(text_emb, shape=(1, 1, 32, 32, 32))
    
    print(f"Generated voxels: {voxels.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

