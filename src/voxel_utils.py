"""
Voxel Utilities
================
Utilities for processing, visualizing, and exporting voxel grids.

Key functions:
1. voxel_to_mesh: Convert voxel grid to 3D mesh
2. make_lego_style: Convert voxels to LEGO brick style
3. visualize_voxels: Interactive 3D visualization
4. export_model: Save as STL/OBJ file for 3D printing
"""

import numpy as np
import torch
import trimesh
import plotly.graph_objects as go
from pathlib import Path


def voxel_to_mesh(voxel_grid, spacing=1.0, blocky=True):
    """
    Convert a voxel grid to a 3D mesh.
    
    Args:
        voxel_grid: 3D numpy array or torch tensor (binary or probability)
        spacing: Size of each voxel in the output mesh
        blocky: If True, creates blocky voxel cubes; if False, uses smooth marching cubes
        
    Returns:
        trimesh.Trimesh: 3D mesh object
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.cpu().numpy()
    
    # Remove batch and channel dimensions if present
    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)
    
    # Ensure binary (0 or 1)
    voxel_grid = (voxel_grid > 0.5).astype(np.float32)
    
    if blocky:
        # Create blocky voxel mesh (individual cubes)
        meshes = []
        for x in range(voxel_grid.shape[0]):
            for y in range(voxel_grid.shape[1]):
                for z in range(voxel_grid.shape[2]):
                    if voxel_grid[x, y, z]:
                        cube = trimesh.creation.box(
                            extents=[spacing, spacing, spacing],
                            transform=trimesh.transformations.translation_matrix(
                                [x * spacing, y * spacing, z * spacing]
                            )
                        )
                        meshes.append(cube)
        
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = trimesh.Trimesh()
    else:
        # Use smooth marching cubes
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
            voxel_grid,
            pitch=spacing
        )
    
    return mesh


def make_lego_style(voxel_grid, brick_size=1, add_studs=True):
    """
    Convert voxel grid to LEGO brick style.
    
    This simplifies the voxel grid and optionally adds LEGO studs on top.
    
    Args:
        voxel_grid: 3D numpy array (binary)
        brick_size: Size of each LEGO brick (1 = 1x1, 2 = 2x2, etc.)
        add_studs: Whether to add LEGO studs on top
        
    Returns:
        trimesh.Trimesh: LEGO-style mesh
    """
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.cpu().numpy()
    
    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)
    
    # Make binary
    voxel_grid = (voxel_grid > 0.5).astype(bool)
    
    # Create mesh for each occupied voxel as a brick
    meshes = []
    
    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z]:
                    # Create a brick (slightly smaller than voxel for gaps)
                    brick = trimesh.creation.box(
                        extents=[0.95, 0.95, 0.95],
                        transform=trimesh.transformations.translation_matrix(
                            [x, y, z]
                        )
                    )
                    meshes.append(brick)
                    
                    # Add stud on top if requested
                    if add_studs and z == voxel_grid.shape[2] - 1:
                        # Check if this is the top layer
                        stud = trimesh.creation.cylinder(
                            radius=0.25,
                            height=0.3,
                            transform=trimesh.transformations.translation_matrix(
                                [x, y, z + 0.65]
                            )
                        )
                        meshes.append(stud)
    
    # Combine all meshes
    if meshes:
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh
    else:
        # Return empty mesh if no voxels
        return trimesh.Trimesh()


def visualize_voxels(voxel_grid, title="3D Voxel Model", show=True, save_path=None):
    """
    Create interactive 3D visualization using Plotly.
    
    Args:
        voxel_grid: 3D numpy array or torch tensor
        title: Title for the plot
        show: Whether to display the plot
        save_path: Path to save HTML file (optional)
        
    Returns:
        plotly.graph_objects.Figure: The figure object
    """
    # Convert to numpy
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.cpu().numpy()
    
    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)
    
    # Make binary
    voxel_grid = (voxel_grid > 0.5).astype(bool)
    
    # Get occupied voxel coordinates
    x, y, z = np.where(voxel_grid)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=z,  # Color by height
            colorscale='Viridis',
            opacity=0.8,
            line=dict(color='white', width=0.5)
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Equal aspect ratio
        ),
        width=800,
        height=800
    )
    
    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved visualization to {save_path}")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def export_model(voxel_grid, output_path, format='stl', lego_style=False, smooth=False):
    """
    Export voxel grid to 3D file format.
    
    Args:
        voxel_grid: 3D numpy array or torch tensor
        output_path: Path to save file
        format: File format ('stl', 'obj', 'ply')
        lego_style: Whether to use LEGO brick style
        smooth: Whether to smooth the mesh (removes blocky voxel appearance)
        
    Returns:
        Path: Path to saved file
    """
    # Convert to mesh with different styles
    if lego_style:
        # LEGO style: blocks with studs
        mesh = make_lego_style(voxel_grid, add_studs=True)
    elif smooth:
        # Smooth style: marching cubes + Laplacian smoothing
        mesh = voxel_to_mesh(voxel_grid, blocky=False)
        import trimesh.smoothing
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=5)
    else:
        # Default: blocky voxel cubes (no studs)
        mesh = voxel_to_mesh(voxel_grid, blocky=True)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export based on format
    if format.lower() == 'stl':
        mesh.export(output_path)
    elif format.lower() == 'obj':
        mesh.export(output_path)
    elif format.lower() == 'ply':
        mesh.export(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Exported model to {output_path}")
    return output_path


def get_voxel_stats(voxel_grid):
    """
    Get statistics about the voxel grid.
    
    Args:
        voxel_grid: 3D numpy array or torch tensor
        
    Returns:
        dict: Statistics dictionary
    """
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.cpu().numpy()
    
    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)
    
    binary_grid = (voxel_grid > 0.5).astype(bool)
    
    total_voxels = binary_grid.size
    occupied_voxels = binary_grid.sum()
    occupancy_rate = occupied_voxels / total_voxels
    
    stats = {
        'shape': voxel_grid.shape,
        'total_voxels': total_voxels,
        'occupied_voxels': int(occupied_voxels),
        'empty_voxels': int(total_voxels - occupied_voxels),
        'occupancy_rate': f"{occupancy_rate*100:.2f}%",
        'min_value': float(voxel_grid.min()),
        'max_value': float(voxel_grid.max()),
        'mean_value': float(voxel_grid.mean())
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    print("Testing voxel utilities...")
    
    # Create a simple test voxel grid (a cube)
    voxel_size = 32
    test_voxel = np.zeros((voxel_size, voxel_size, voxel_size))
    
    # Create a simple shape (pyramid)
    for z in range(16):
        size = 16 - z
        start = 8 + z // 2
        test_voxel[start:start+size, start:start+size, z] = 1
    
    print("\nVoxel Statistics:")
    stats = get_voxel_stats(test_voxel)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test mesh conversion
    print("\nConverting to mesh...")
    mesh = voxel_to_mesh(test_voxel)
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    
    # Test LEGO style conversion
    print("\nConverting to LEGO style...")
    lego_mesh = make_lego_style(test_voxel, add_studs=True)
    print(f"  Vertices: {len(lego_mesh.vertices)}")
    print(f"  Faces: {len(lego_mesh.faces)}")
    
    print("\nUtilities test complete!")

