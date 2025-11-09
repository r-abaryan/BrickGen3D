"""
Quick Test - Try Models With/Without Training
=========================================
Test diffusion and GNN models with random weights (untrained).
"""

import torch
from src.text_encoder import TextEncoder
from src.diffusion_model import UNet3D, VoxelDiffusion
from src.gnn_model import MeshGNN
from src.voxel_utils import visualize_voxels, export_model


def test_diffusion():
    """Test diffusion model (untrained)."""
    print("\n" + "="*60)
    print("Testing Diffusion Model (Untrained)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize
    text_encoder = TextEncoder()
    model = UNet3D(text_dim=384, base_channels=32).to(device)
    diffusion = VoxelDiffusion(model, num_steps=50, device=device)
    
    # Generate
    text = "a cube"
    print(f"\nText: '{text}'")
    
    text_emb = text_encoder.encode(text).to(device)
    voxels = diffusion.sample(text_emb, shape=(1, 1, 32, 32, 32))
    
    print(f"Generated: {voxels.shape}")
    print(f"Occupied: {(voxels > 0.5).sum().item()} / {32**3} voxels")
    
    # Visualize
    visualize_voxels(voxels, title="Diffusion (Untrained)", show=True)
    
    # Export
    export_model(voxels, "outputs/diffusion_test.stl", lego_style=False)
    print("✓ Saved to outputs/diffusion_test.stl")


def test_gnn():
    """Test GNN model (untrained)."""
    print("\n" + "="*60)
    print("Testing GNN Model (Untrained)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize
    text_encoder = TextEncoder()
    model = MeshGNN(text_dim=384, hidden_dim=256, num_layers=4).to(device)
    
    # Generate
    text = "a sphere"
    print(f"\nText: '{text}'")
    
    text_emb = text_encoder.encode(text).to(device)
    vertices = model(text_emb)
    
    print(f"Generated: {vertices.shape[1]} vertices")
    
    # Convert to point cloud for visualization
    import numpy as np
    vertices_np = vertices[0].detach().cpu().numpy()
    
    # Create simple voxel representation from points
    voxels = np.zeros((32, 32, 32))
    for v in vertices_np:
        # Scale to voxel grid
        x = int((v[0] + 1) * 16)
        y = int((v[1] + 1) * 16)
        z = int((v[2] + 1) * 16)
        if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
            voxels[x, y, z] = 1
    
    print(f"Occupied: {voxels.sum()} voxels")
    
    # Visualize
    visualize_voxels(voxels, title="GNN (Untrained)", show=True)
    
    # Export
    export_model(voxels, "outputs/gnn_test.stl", lego_style=False)
    print("✓ Saved to outputs/gnn_test.stl")


def test_open3d():
    """Test Open3D utilities."""
    print("\n" + "="*60)
    print("Testing Open3D Utilities")
    print("="*60)
    
    try:
        import open3d as o3d
        from src.open3d_utils import voxel_to_mesh, process_mesh
        
        # Create test voxels (cube)
        import numpy as np
        voxels = np.zeros((32, 32, 32))
        voxels[10:22, 10:22, 10:22] = 1
        
        print("\nConverting voxels to mesh...")
        mesh = voxel_to_mesh(voxels, voxel_size=1.0)
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        print("Processing mesh (merging vertices, smoothing)...")
        mesh = process_mesh(mesh, smooth=True, simplify=True, merge_vertices=True)
        print(f"Processed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        # Save
        import open3d as o3d
        o3d.io.write_triangle_mesh("outputs/open3d_test.stl", mesh)
        print("✓ Saved to outputs/open3d_test.stl")
        
    except ImportError:
        print("❌ Open3D not installed. Run: pip install open3d")


def main():
    """Run all tests."""
    print("="*60)
    print("BrickGen3D - Model Testing (Untrained)")
    print("="*60)
    print("\nNote: Models are UNTRAINED - output will be random!")
    print("This is just to test if everything works.\n")
    
    choice = input("Test: [1] Diffusion, [2] GNN, [3] Open3D, [4] All: ")
    
    if choice == '1':
        test_diffusion()
    elif choice == '2':
        test_gnn()
    elif choice == '3':
        test_open3d()
    elif choice == '4':
        test_open3d()
        test_diffusion()
        test_gnn()
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

