# BrickGen3D: AI-Assisted LEGO-Style 3D Build Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Generate 3D models from text descriptions using AI, with optional LEGO brick styling!

```
"a large blue cube" → 🤖 → 🧱 3D Model
```

## Features

- **Text-to-3D Generation**: Convert natural language to 3D voxel models
- **Lightweight**: ~10M parameters, trains in minutes on CPU
- **LEGO-Style Output**: Generate models with LEGO brick aesthetics
- **Multiple Formats**: STL, OBJ, PLY export
- **Multiple Models**: Base, Diffusion, GNN, Open3D variants

## Quick Start

### Installation

```bash
pip install -r requirements.txt
python train/train.py  # Train base model (~15 min CPU)
```

### Generate Models

```bash
# Base generator
python generate.py --text "a cube" --output outputs/cube.stl

# Styles: --lego_style, --smooth
python generate.py --text "a sphere" --output sphere_lego.stl --lego_style

# Other models
python generate_diffusion.py --text "a cube"
python generate_gnn.py --text "a cube"
python generate_open3d.py --text "a cube" --smooth
```

**Flags**: `--text`, `--output`, `--threshold`, `--checkpoint`, `--lego_style`, `--smooth`, `--no_show`

## How It Works

```
text → (MiniLM encoder) → 384-D vector → (MLP + 3D Conv) → 32³ voxels → STL/OBJ/PLY
```

- **Base Model**: Text encoder + 3D convolutional decoder (`src/voxel_generator.py`)
- **Diffusion**: Iterative denoising with timestep embeddings (`src/diffusion_model.py`)
- **GNN**: Graph convolutions for mesh deformation (`src/gnn_model.py`)
- **Export**: Blocky cubes, LEGO studs, or smooth marching cubes (`src/voxel_utils.py`)

## Project Structure

```
src/              # Core modules (text_encoder, voxel_generator, diffusion_model, gnn_model)
train/            # Training scripts (train.py, train_diffusion.py, train_gnn.py)
generate*.py      # Generation scripts
checkpoints/      # Saved models
outputs/          # Generated files
other/            # Educational docs
```


## Citation & License

```bibtex
@software{BrickGen3D,
  title={Generate 3D models from text descriptions using AI},
  author={Abaryan},
  year={2025},
  url={https://github.com/r-abaryan/BrickGen3D}
}
```