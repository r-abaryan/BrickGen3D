"""
Simple Dataset for Text-to-3D
==============================
This module creates a simple synthetic dataset for training.

Since we don't have a large dataset of text-3D pairs, we'll create
simple geometric shapes with corresponding descriptions.

Shapes included:
- Cube, Sphere, Pyramid, Cylinder, etc.
- Different sizes and positions
- Combined shapes

This is a STARTING POINT. For real applications, you'd use:
- ShapeNet (55 object categories)
- Objaverse (800K+ 3D models)
- Custom 3D model collections
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SimpleShapeDataset(Dataset):
    """
    Generates simple 3D shapes with text descriptions.
    
    This is a procedural dataset - shapes are generated on-the-fly,
    so it never runs out of training samples!
    """
    
    def __init__(self, num_samples=1000, voxel_size=32):
        """
        Args:
            num_samples: Number of samples per epoch
            voxel_size: Size of voxel grid (32x32x32)
        """
        self.num_samples = num_samples
        self.voxel_size = voxel_size
        
        # Shape generation functions
        self.shape_functions = {
            'cube': self._create_cube,
            'sphere': self._create_sphere,
            'pyramid': self._create_pyramid,
            'cylinder': self._create_cylinder,
            'cone': self._create_cone,
        }
        
        # Size descriptors
        self.sizes = ['tiny', 'small', 'medium', 'large']
        
        # Colors (for description variety)
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a random shape with description.
        
        Returns:
            tuple: (text_description, voxel_grid)
        """
        # Random seed based on index for reproducibility
        random.seed(idx)
        np.random.seed(idx)
        
        # Choose random shape and properties
        shape_name = random.choice(list(self.shape_functions.keys()))
        size_desc = random.choice(self.sizes)
        color = random.choice(self.colors)
        
        # Map size description to actual size
        size_map = {'tiny': 6, 'small': 10, 'medium': 14, 'large': 18}
        size = size_map[size_desc]
        
        # Generate shape
        voxel_grid = self.shape_functions[shape_name](size)
        
        # Generate text description
        # Format: "a [size] [color] [shape]"
        text = f"a {size_desc} {color} {shape_name}"
        
        # Convert to torch tensor
        voxel_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0)  # Add channel dim
        
        return text, voxel_tensor
    
    def _create_cube(self, size):
        """Create a cube centered in the grid."""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        center = self.voxel_size // 2
        half_size = size // 2
        
        voxel[
            center - half_size:center + half_size,
            center - half_size:center + half_size,
            center - half_size:center + half_size
        ] = 1
        
        return voxel
    
    def _create_sphere(self, size):
        """Create a sphere centered in the grid."""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        center = self.voxel_size // 2
        radius = size // 2
        
        # Use distance from center
        for x in range(self.voxel_size):
            for y in range(self.voxel_size):
                for z in range(self.voxel_size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                    if dist <= radius:
                        voxel[x, y, z] = 1
        
        return voxel
    
    def _create_pyramid(self, size):
        """Create a pyramid."""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        center = self.voxel_size // 2
        base_half_size = size // 2
        height = size
        
        for z in range(height):
            # Size decreases linearly with height
            layer_size = int(base_half_size * (1 - z / height))
            if layer_size > 0:
                voxel[
                    center - layer_size:center + layer_size,
                    center - layer_size:center + layer_size,
                    z
                ] = 1
        
        return voxel
    
    def _create_cylinder(self, size):
        """Create a vertical cylinder."""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        center = self.voxel_size // 2
        radius = size // 2
        height = size
        
        for x in range(self.voxel_size):
            for y in range(self.voxel_size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist <= radius:
                    voxel[x, y, :height] = 1
        
        return voxel
    
    def _create_cone(self, size):
        """Create a cone."""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        center = self.voxel_size // 2
        base_radius = size // 2
        height = size
        
        for z in range(height):
            # Radius decreases linearly with height
            layer_radius = base_radius * (1 - z / height)
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    if dist <= layer_radius:
                        voxel[x, y, z] = 1
        
        return voxel


class CombinedShapeDataset(SimpleShapeDataset):
    """
    Extended dataset that can combine multiple shapes.
    
    Examples:
    - "a small cube and a large sphere"
    - "two medium pyramids"
    - "a tower of cubes"
    """
    
    def __getitem__(self, idx):
        """Generate single or combined shapes."""
        random.seed(idx)
        np.random.seed(idx)
        
        # 70% chance single shape, 30% chance combined
        if random.random() < 0.7:
            return super().__getitem__(idx)
        
        # Generate 2 shapes
        shape1 = random.choice(list(self.shape_functions.keys()))
        shape2 = random.choice(list(self.shape_functions.keys()))
        
        size1 = random.choice([6, 8, 10])  # Smaller for combined shapes
        size2 = random.choice([6, 8, 10])
        
        color1 = random.choice(self.colors)
        color2 = random.choice(self.colors)
        
        # Create voxel grid with both shapes
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        
        # Place first shape (left side)
        offset1 = self.voxel_size // 4
        temp_center = self.voxel_size // 2
        self.voxel_size = offset1 * 2  # Temporarily adjust for positioning
        voxel1 = self.shape_functions[shape1](size1)
        self.voxel_size = 32  # Reset
        
        # Add first shape
        voxel[:voxel1.shape[0], :voxel1.shape[1], :voxel1.shape[2]] += voxel1
        
        # Place second shape (right side)
        offset2 = 3 * self.voxel_size // 4
        voxel2 = self.shape_functions[shape2](size2)
        
        # Simple positioning (just combine, may overlap)
        voxel = np.maximum(voxel, voxel2)  # Union of shapes
        
        # Generate description
        size_desc1 = random.choice(['small', 'tiny'])
        size_desc2 = random.choice(['small', 'tiny'])
        text = f"a {size_desc1} {color1} {shape1} and a {size_desc2} {color2} {shape2}"
        
        voxel_tensor = torch.from_numpy(voxel).float().unsqueeze(0)
        return text, voxel_tensor


# Testing
if __name__ == "__main__":
    print("Testing SimpleShapeDataset...")
    
    # Create dataset
    dataset = SimpleShapeDataset(num_samples=10, voxel_size=32)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a few samples
    for i in range(3):
        text, voxel = dataset[i]
        occupied = (voxel > 0.5).sum().item()
        print(f"\nSample {i}:")
        print(f"  Text: '{text}'")
        print(f"  Voxel shape: {voxel.shape}")
        print(f"  Occupied voxels: {occupied}")
    
    print("\n\nTesting CombinedShapeDataset...")
    dataset2 = CombinedShapeDataset(num_samples=10, voxel_size=32)
    
    for i in range(3):
        text, voxel = dataset2[i]
        occupied = (voxel > 0.5).sum().item()
        print(f"\nSample {i}:")
        print(f"  Text: '{text}'")
        print(f"  Occupied voxels: {occupied}")

