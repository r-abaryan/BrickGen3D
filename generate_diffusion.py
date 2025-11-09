"""
Generate 3D models using Diffusion Model
=========================================
Usage: python generate_diffusion.py --text "a red pyramid" --output test.stl
"""

import torch
import argparse
from pathlib import Path

from src.text_encoder import TextEncoder
from src.diffusion_model import UNet3D, VoxelDiffusion
from src.voxel_utils import visualize_voxels, export_model, get_voxel_stats


def generate(
    text,
    checkpoint='checkpoints/diffusion_final.pth',
    output='outputs/diffusion_output.stl',
    threshold=0.5,
    lego_style=False,
    show=True,
    device=None
):
    """Generate 3D model using diffusion."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Text: '{text}'")
    
    # Initialize
    text_encoder = TextEncoder()
    model = UNet3D(text_dim=384, base_channels=32).to(device)
    diffusion = VoxelDiffusion(model, num_steps=50, device=device)
    
    # Load checkpoint if exists
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state if isinstance(state, dict) and 'model_state_dict' not in state else state['model_state_dict'])
        print(f"✓ Loaded trained model from {checkpoint}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint}")
        print("  Using untrained model (random output)")
    
    # Generate
    print("\nGenerating (50 diffusion steps)...")
    with torch.no_grad():
        text_emb = text_encoder.encode(text).to(device)
        voxels = diffusion.sample(text_emb, shape=(1, 1, 32, 32, 32))
    
    # Apply threshold
    voxels_binary = (voxels > threshold).cpu().numpy()
    
    # Stats
    print("\nVoxel Statistics:")
    stats = get_voxel_stats(voxels_binary)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Visualize
    if show:
        visualize_voxels(voxels_binary, title=f"Diffusion: {text}", show=True)
    
    # Export
    print(f"\nExporting to {output}...")
    export_model(voxels_binary, output, format=Path(output).suffix[1:], lego_style=lego_style)
    print("✓ Done!")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D models using diffusion')
    parser.add_argument('--text', '-t', type=str, required=True, help='Text description')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/diffusion_final.pth')
    parser.add_argument('--output', '-o', type=str, default='outputs/diffusion_output.stl')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--lego_style', '-l', action='store_true')
    parser.add_argument('--no_show', action='store_true')
    
    args = parser.parse_args()
    
    generate(
        text=args.text,
        checkpoint=args.checkpoint,
        output=args.output,
        threshold=args.threshold,
        lego_style=args.lego_style,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()

