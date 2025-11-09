"""
GNN-based Mesh Generator
========================
Uses Graph Neural Networks to generate mesh directly (not voxels).

Output: Vertices + Faces (proper mesh topology)
Advantage: Smoother surfaces, better topology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Graph convolution layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, node_features, adjacency):
        """
        Args:
            node_features: (batch, num_nodes, in_features)
            adjacency: (batch, num_nodes, num_nodes) adjacency matrix
        """
        # Aggregate neighbor features
        aggregated = torch.bmm(adjacency, node_features)
        
        # Transform
        output = self.linear(aggregated)
        return F.relu(output)


class MeshGNN(nn.Module):
    """
    GNN-based mesh generator.
    
    Process:
    1. Start with template mesh (e.g., icosphere)
    2. Use GNN to deform vertices based on text
    3. Output deformed mesh
    """
    def __init__(self, text_dim=384, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # Text encoder projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output: vertex displacements (x, y, z)
        self.output = nn.Linear(hidden_dim, 3)
        
        # Template mesh (icosphere with 162 vertices)
        self.register_buffer('template_vertices', self._create_icosphere(subdivisions=2))
        self.register_buffer('template_adjacency', self._compute_adjacency(self.template_vertices))
    
    def _create_icosphere(self, subdivisions=2):
        """Create icosphere template."""
        # Golden ratio
        phi = (1 + 5**0.5) / 2
        
        # Icosahedron vertices
        vertices = torch.tensor([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
        ], dtype=torch.float32)
        
        # Normalize
        vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)
        
        # TODO: Add subdivision for higher resolution
        return vertices
    
    def _compute_adjacency(self, vertices):
        """Compute adjacency matrix (simplified: k-nearest neighbors)."""
        num_verts = vertices.shape[0]
        
        # Compute pairwise distances
        dist = torch.cdist(vertices, vertices)
        
        # k-nearest neighbors (k=6 for mesh-like structure)
        k = 6
        _, indices = torch.topk(dist, k=k+1, largest=False, dim=1)
        
        # Create adjacency matrix
        adjacency = torch.zeros(num_verts, num_verts)
        for i in range(num_verts):
            adjacency[i, indices[i, 1:]] = 1  # Skip self
        
        # Normalize
        degree = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (degree + 1e-6)
        
        return adjacency
    
    def forward(self, text_emb):
        """
        Args:
            text_emb: (batch, text_dim)
        Returns:
            vertices: (batch, num_vertices, 3)
        """
        batch_size = text_emb.shape[0]
        num_verts = self.template_vertices.shape[0]
        
        # Project text to node features
        text_features = self.text_proj(text_emb)  # (batch, hidden_dim)
        
        # Broadcast to all vertices
        node_features = text_features.unsqueeze(1).expand(batch_size, num_verts, -1)
        
        # Expand adjacency for batch
        adjacency = self.template_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            node_features = gnn(node_features, adjacency)
        
        # Predict vertex displacements
        displacements = self.output(node_features)
        
        # Apply displacements to template
        template = self.template_vertices.unsqueeze(0).expand(batch_size, -1, -1)
        vertices = template + displacements
        
        return vertices
    
    def get_faces(self):
        """Return face indices (fixed for template)."""
        # TODO: Return proper face indices based on icosphere topology
        # For now, return empty (visualization will use point cloud)
        return torch.tensor([], dtype=torch.long)


class MeshGNNGenerator:
    """Wrapper for mesh generation."""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def generate(self, text_emb):
        """Generate mesh from text embedding."""
        self.model.eval()
        vertices = self.model(text_emb.to(self.device))
        faces = self.model.get_faces()
        return vertices, faces


# Example usage
if __name__ == "__main__":
    model = MeshGNN(text_dim=384, hidden_dim=256, num_layers=4)
    
    # Test
    text_emb = torch.randn(2, 384)
    vertices = model(text_emb)
    
    print(f"Generated mesh:")
    print(f"  Batch size: {vertices.shape[0]}")
    print(f"  Vertices: {vertices.shape[1]}")
    print(f"  Dimensions: {vertices.shape[2]}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

