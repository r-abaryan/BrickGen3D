# BrickGen3D: AI-Assisted LEGO-Style 3D Build Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Generate 3D models from text descriptions using AI, with optional LEGO brick styling!

```
"a large blue cube" → 🤖 → 🧱 3D Model
```

## Features

- **Text-to-3D Generation**: Convert natural language descriptions to 3D voxel models
- **Lightweight Model**: Only ~10M parameters, trains in minutes on CPU
- **LEGO-Style Output**: Generate models with LEGO brick aesthetics (with studs!)
- **Multiple Export Formats**: STL, OBJ, PLY for 3D printing or viewing
- **Interactive Visualization**: View your creations in 3D before exporting
- **Easy to Use**: Simple command-line interface
- **Fully Explained**: Every part of the code is documented and explained

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Training](#training)
- [Generating Models](#generating-models)
- [Architecture Details](#architecture-details)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Step 1: Clone or Download

```bash
pip install -r requirements.txt     # PyTorch, trimesh, etc.

# Base model (~15 min CPU)
python train/train.py

# Optional experiments
python train/train_diffusion.py     # needs 150+ epochs for best quality
python train/train_gnn.py
```

---

## Generate

```bash
# Base generator
python generate.py --text "a cube" --output outputs/cube.stl

# Mesh styles
python generate.py --text "a sphere" --output sphere_blocky.stl
python generate.py --text "a sphere" --output sphere_lego.stl --lego_style
python generate.py --text "a sphere" --output sphere_smooth.stl --smooth

# Other emitters
python generate_diffusion.py --text "a cube"
python generate_gnn.py --text "a cube"
python generate_open3d.py --text "a cube" --smooth
```

Shared flags: `--text`, `--output`, `--threshold`, `--checkpoint`, `--lego_style`, `--smooth`, `--no_show`.

---

## How It Works (TL;DR)

```
text --(MiniLM encoder)--> 384-D vector
      --(MLP + 3D Conv decoder)--> 32x32x32 voxels
      --(voxel_utils)--> STL/OBJ/PLY
```

- Dataset: procedural cubes, spheres, pyramids (`src/dataset.py`)
- Loss: voxel-wise BCE (`train/train.py`)
- Export styles: blocky cubes / LEGO studs / smooth marching cubes (`src/voxel_utils.py`)
- Diffusion: iterative denoising with timestep embeddings (`src/diffusion_model.py`)
- GNN: deform template mesh via graph convolutions (`src/gnn_model.py`)

---

## Repo Map

```
BrickGen3D/
├── src/                      # Core modules
│   ├── text_encoder.py       # Text → embeddings
│   ├── voxel_generator.py    # Basic model
│   ├── diffusion_model.py   # Diffusion model
│   ├── gnn_model.py         # GNN model
│   ├── dataset.py           # Training data
│   ├── voxel_utils.py       # Basic utilities
│   └── open3d_utils.py      # Open3D utilities
├── train/                    # Training scripts
│   ├── train.py             # Basic model
│   ├── train_diffusion.py   # Diffusion
│   └── train_gnn.py         # GNN
├── generate.py              # Basic generation
├── generate_diffusion.py    # Diffusion generation
├── generate_gnn.py          # GNN generation
├── generate_open3d.py       # Open3D generation
├── test_models.py           # Test scripts
├── checkpoints/             # Saved models
└── outputs/                 # Generated files
```

## Key Concepts

- **Voxel**: 3D pixel (like Minecraft blocks)
- **Embedding**: Text → numerical representation
- **Training**: Model learns text → shape mapping
- **Generation**: Text → 3D model

See `other/` folder for detailed educational docs.


## Contributing

This is an educational project! Feel free to:
- Experiment with different architectures
- Try real datasets
- Add new features
- Improve documentation
- Share your results!

## Advanced Models

### Open3D
```bash
pip install open3d
python generate_open3d.py --text "a cube" --smooth
```

train/                   # Training entry points
generate*.py             # Inference scripts (base, diffusion, GNN, Open3D)
other/                   # Long-form learning docs (optional)
checkpoints/, outputs/   # Generated weights + meshes
```

---

## Tips & Troubleshooting

- Checkpoint missing? → run `python train/train.py`
- Empty mesh? → lower `--threshold` (e.g., 0.3)
- BatchNorm error? → ensure inference scripts call `model.eval()` (already handled)
- Trimesh import issue? → `pip install trimesh --no-deps && pip install numpy networkx`
- Diffusion stuck generating cubes? → train longer (150+ epochs) or use base model

---

## Learn the Concepts

| Topic | File |
|-------|------|
| Text embeddings | `src/text_encoder.py` |
| Voxel decoder (MLP + 3D Conv) | `src/voxel_generator.py` |
| Diffusion blocks + timestep encoding | `src/diffusion_model.py` |
| GNN mesh deformation | `src/gnn_model.py` |
| Marching cubes / LEGO studs | `src/voxel_utils.py` |

Long-form explanations live in `other/` (`START_HERE.md`, `CONCEPTS.md`, etc.).

---

## Citation & License

```bibtex
@software{BrickGen3D,
  title={Generate 3D models from text descriptions using AI},
  author={Abaryan},
  year={2025},
  url={https://github.com/r-abaryan/BrickGen3D}
}
```

MIT License. Contributions welcome.

**Happy building! 🧱**