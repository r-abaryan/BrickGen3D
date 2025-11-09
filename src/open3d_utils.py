"""
Open3D Utilities - Professional 3D Processing
==============================================
Replaces trimesh with Open3D for better mesh operations.

Install: pip install open3d
"""

import numpy as np
import torch
import open3d as o3d


def voxel_to_mesh(voxel_grid, voxel_size=1.0):
    """Convert voxel grid to mesh using Open3D."""
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.cpu().numpy()
    
    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)
    
    voxel_grid = (voxel_grid > 0.5).astype(bool)
    
    # Convert to mesh - create boxes for each voxel
    meshes = []
    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z]:
                    box = o3d.geometry.TriangleMesh.create_box(
                        width=voxel_size, height=voxel_size, depth=voxel_size
                    )
                    box.translate([x * voxel_size, y * voxel_size, z * voxel_size])
                    meshes.append(box)
    
    # Merge all boxes
    if len(meshes) == 0:
        return o3d.geometry.TriangleMesh()
    
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh += m
    
    mesh.compute_vertex_normals()
    
    return mesh


def process_mesh(mesh, smooth=True, simplify=False, target_faces=5000, merge_vertices=True):
    """Post-process mesh with smoothing and simplification."""
    if merge_vertices:
        # Remove duplicate vertices
        mesh = mesh.merge_close_vertices(eps=0.0001)
    
    if smooth:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
    
    if simplify:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    
    mesh.compute_vertex_normals()
    return mesh


def visualize(mesh, window_name="Open3D Viewer"):
    """Interactive 3D visualization."""
    o3d.visualization.draw_geometries([mesh], window_name=window_name)


def save_mesh(mesh, filepath):
    """Save mesh to file."""
    o3d.io.write_triangle_mesh(str(filepath), mesh)
    return filepath


# Example usage
if __name__ == "__main__":
    # Test with simple voxel grid
    test_voxels = np.zeros((32, 32, 32))
    test_voxels[10:22, 10:22, 10:22] = 1
    
    mesh = voxel_to_mesh(test_voxels)
    mesh = process_mesh(mesh, smooth=True)
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

