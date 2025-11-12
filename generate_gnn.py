"""
Generate 3D models using GNN Model
===================================
Usage: python generate_gnn.py --text "a sphere" --output test.stl
"""

import torch
import argparse
import numpy as np
from pathlib import Path

from src.text_encoder import TextEncoder
from src.gnn_model import MeshGNN
from src.voxel_utils import visualize_voxels, export_model


def generate(
    text,
    checkpoint='checkpoints/gnn_final.pth',
    output='outputs/gnn_output.stl',
    lego_style=False,
    show=True,
    device=None
):
    """Generate 3D model using GNN."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Text: '{text}'")
    
    # Initialize
    text_encoder = TextEncoder()
    model = MeshGNN(text_dim=384, hidden_dim=256, num_layers=4).to(device)
    
    # Load checkpoint if exists
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state if isinstance(state, dict) and 'model_state_dict' not in state else state['model_state_dict'])
        print(f"✓ Loaded trained model from {checkpoint}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint}")
        print("  Using untrained model (random output)")
    
    # Set to eval mode
    model.eval()
    
    # Generate
    print("\nGenerating mesh vertices...")
    with torch.no_grad():
        text_emb = text_encoder.encode(text).to(device)
        vertices = model(text_emb)
    
    vertices_np = vertices[0].detach().cpu().numpy()
    print(f"Generated: {len(vertices_np)} vertices")
    
    # Convert to voxels for visualization/export
    voxels = np.zeros((32, 32, 32))
    for v in vertices_np:
        x = int((v[0] + 1) * 16)
        y = int((v[1] + 1) * 16)
        z = int((v[2] + 1) * 16)
        if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
            voxels[x, y, z] = 1
    
    print(f"Occupied voxels: {int(voxels.sum())}")
    
    # Visualize
    if show:
        visualize_voxels(voxels, title=f"GNN: {text}", show=True)
    
    # Export
    print(f"\nExporting to {output}...")
    export_model(voxels, output, format=Path(output).suffix[1:], lego_style=lego_style)
    print("✓ Done!")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D models using GNN')
    parser.add_argument('--text', '-t', type=str, required=True, help='Text description')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/gnn_final.pth')
    parser.add_argument('--output', '-o', type=str, default='outputs/gnn_output.stl')
    parser.add_argument('--lego_style', '-l', action='store_true')
    parser.add_argument('--no_show', action='store_true')
    
    args = parser.parse_args()
    
    generate(
        text=args.text,
        checkpoint=args.checkpoint,
        output=args.output,
        lego_style=args.lego_style,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()

