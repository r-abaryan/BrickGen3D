"""
Generate 3D models using Open3D utilities
==========================================
Uses basic model but with Open3D for better mesh processing.
Usage: python generate_open3d.py --text "a cube" --output test.stl
"""

import torch
import argparse
from pathlib import Path

from src.text_encoder import TextEncoder
from src.voxel_generator import VoxelGenerator


def generate(
    text,
    checkpoint='checkpoints/best_model.pth',
    output='outputs/open3d_output.stl',
    threshold=0.5,
    smooth=True,
    simplify=False,
    show_viewer=False,
    device=None
):
    """Generate 3D model with Open3D processing."""
    try:
        import open3d as o3d
        from src.open3d_utils import voxel_to_mesh, process_mesh, save_mesh
    except ImportError:
        print("❌ Open3D not installed. Run: pip install open3d")
        return
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Text: '{text}'")
    
    # Initialize
    text_encoder = TextEncoder()
    model = VoxelGenerator(text_dim=384, voxel_size=32, hidden_dim=512).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"✓ Loaded trained model")
    else:
        print(f"⚠ No checkpoint found - using untrained model")
    
    # Generate
    print("\nGenerating voxels...")
    with torch.no_grad():
        text_emb = text_encoder.encode(text).to(device)
        voxels = model.generate(text_emb, threshold=threshold)
    
    voxels_np = voxels.cpu().numpy()
    print(f"Occupied: {int((voxels_np > 0.5).sum())} / {32**3} voxels")
    
    # Convert to mesh with Open3D
    print("\nConverting to mesh with Open3D...")
    mesh = voxel_to_mesh(voxels_np, voxel_size=1.0)
    print(f"  Initial: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # Process
    print("Processing mesh...")
    mesh = process_mesh(mesh, smooth=smooth, simplify=simplify, merge_vertices=True)
    print(f"  Processed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # Visualize
    if show_viewer:
        print("\nOpening Open3D viewer...")
        o3d.visualization.draw_geometries([mesh], window_name=f"Open3D: {text}")
    
    # Save
    print(f"\nSaving to {output}...")
    save_mesh(mesh, output)
    print("✓ Done!")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D models with Open3D')
    parser.add_argument('--text', '-t', type=str, required=True, help='Text description')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output', '-o', type=str, default='outputs/open3d_output.stl')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing')
    parser.add_argument('--simplify', action='store_true', help='Simplify mesh')
    parser.add_argument('--show', action='store_true', help='Show Open3D viewer')
    
    args = parser.parse_args()
    
    generate(
        text=args.text,
        checkpoint=args.checkpoint,
        output=args.output,
        threshold=args.threshold,
        smooth=args.smooth,
        simplify=args.simplify,
        show_viewer=args.show
    )


if __name__ == "__main__":
    main()

