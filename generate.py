"""
Generation Script for BrickGen3D
=================================
This script generates 3D models from text descriptions.

Usage:
    python generate.py --text "a large blue cube"
    python generate.py --text "a small red pyramid" --lego_style --output my_model.stl

The script will:
1. Load the trained model
2. Encode your text description
3. Generate a 3D voxel model
4. Visualize it interactively
5. Export to file (STL, OBJ, etc.)
"""

import torch
import argparse
from pathlib import Path
import sys

from src.text_encoder import TextEncoder
from src.voxel_generator import VoxelGenerator
from src.voxel_utils import (
    visualize_voxels,
    export_model,
    get_voxel_stats,
    make_lego_style,
    voxel_to_mesh
)


class BrickGen3D:
    """
    Main class for generating 3D models from text.
    """
    
    def __init__(self, checkpoint_path='checkpoints/best_model.pth', device=None):
        """
        Initialize the generator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load text encoder
        print("Loading text encoder...")
        self.text_encoder = TextEncoder()
        
        # Initialize voxel generator
        print("Loading voxel generator...")
        self.model = VoxelGenerator(
            text_dim=384,
            voxel_size=32,
            hidden_dim=512
        ).to(self.device)
        
        # Load trained weights
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"\n⚠ Warning: Checkpoint not found at {checkpoint_path}")
            print("The model is UNTRAINED and will generate random outputs.")
            print("Please run 'python train.py' first to train the model.\n")
            self.is_trained = False
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded trained model from epoch {checkpoint['epoch']}")
            self.is_trained = True
        
        self.model.eval()  # Set to evaluation mode
    
    def generate(self, text, threshold=0.5):
        """
        Generate 3D voxel model from text description.
        
        Args:
            text: Text description (e.g., "a large blue cube")
            threshold: Voxel occupancy threshold (0.0 to 1.0)
            
        Returns:
            numpy array: Binary voxel grid
        """
        print(f"\nGenerating 3D model for: '{text}'")
        
        # Encode text
        with torch.no_grad():
            text_embedding = self.text_encoder.encode(text).to(self.device)
            
            # Generate voxels
            voxel_prob = self.model(text_embedding)
            
            # Apply threshold to get binary voxels
            voxel_binary = (voxel_prob > threshold).float()
        
        # Convert to numpy
        voxel_numpy = voxel_binary.cpu().numpy()
        
        return voxel_numpy
    
    def generate_and_visualize(
        self,
        text,
        threshold=0.5,
        lego_style=False,
        show=True,
        save_html=None,
        export_path=None,
        export_format='stl'
    ):
        """
        Generate and visualize a 3D model.
        
        Args:
            text: Text description
            threshold: Voxel occupancy threshold
            lego_style: Whether to use LEGO brick style
            show: Whether to show interactive visualization
            save_html: Path to save HTML visualization
            export_path: Path to export 3D model file
            export_format: Export format ('stl', 'obj', 'ply')
        """
        # Generate voxels
        voxels = self.generate(text, threshold)
        
        # Print statistics
        print("\nVoxel Statistics:")
        stats = get_voxel_stats(voxels)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Visualize
        if show or save_html:
            print("\nCreating visualization...")
            visualize_voxels(
                voxels,
                title=f"BrickGen3D: {text}",
                show=show,
                save_path=save_html
            )
        
        # Export to file
        if export_path:
            print(f"\nExporting to {export_path}...")
            export_model(
                voxels,
                export_path,
                format=export_format,
                lego_style=lego_style
            )
            print(f"✓ Model exported successfully!")
            print(f"  You can now 3D print or view this file in any 3D software")
        
        return voxels


def main():
    """Command-line interface for generation."""
    parser = argparse.ArgumentParser(
        description='Generate 3D models from text using BrickGen3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --text "a large blue cube"
  python generate.py --text "a small red pyramid" --lego_style
  python generate.py --text "a medium sphere" --output sphere.stl
  python generate.py --text "a green cylinder" --no_show --output model.obj
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        required=True,
        help='Text description of the 3D model to generate'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Voxel occupancy threshold (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--lego_style', '-l',
        action='store_true',
        help='Use LEGO brick style with studs'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (e.g., model.stl, model.obj)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='stl',
        choices=['stl', 'obj', 'ply'],
        help='Output file format (default: stl)'
    )
    
    parser.add_argument(
        '--save_html',
        type=str,
        help='Save interactive HTML visualization'
    )
    
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not show interactive visualization'
    )
    
    args = parser.parse_args()
    
    # Create generator
    print("="*60)
    print("BrickGen3D - Text to 3D Generator")
    print("="*60)
    
    generator = BrickGen3D(checkpoint_path=args.checkpoint)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate filename from text
        safe_text = "".join(c for c in args.text if c.isalnum() or c == ' ')
        safe_text = safe_text.replace(' ', '_')[:50]
        output_path = f"outputs/{safe_text}.{args.format}"
    
    # Generate and visualize
    generator.generate_and_visualize(
        text=args.text,
        threshold=args.threshold,
        lego_style=args.lego_style,
        show=not args.no_show,
        save_html=args.save_html,
        export_path=output_path,
        export_format=args.format
    )
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    if args.output or not args.no_show:
        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()

