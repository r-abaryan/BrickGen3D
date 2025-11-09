# BrickGen3D: AI-Assisted LEGO-Style 3D Build Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Generate 3D models from text descriptions using AI, with optional LEGO brick styling!

```
"a large blue cube" → 🤖 → 🧱 3D Model
```

## 🌟 Features

- **Text-to-3D Generation**: Convert natural language descriptions to 3D voxel models
- **Lightweight Model**: Only ~10M parameters, trains in minutes on CPU
- **LEGO-Style Output**: Generate models with LEGO brick aesthetics (with studs!)
- **Multiple Export Formats**: STL, OBJ, PLY for 3D printing or viewing
- **Interactive Visualization**: View your creations in 3D before exporting
- **Easy to Use**: Simple command-line interface
- **Fully Explained**: Every part of the code is documented and explained

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Training](#training)
- [Generating Models](#generating-models)
- [Architecture Details](#architecture-details)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Step 1: Clone or Download

```bash
cd BrickGen3D
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows users**: If you encounter issues with `trimesh`, you may need to install it separately:

```bash
pip install trimesh --no-deps
pip install numpy networkx
```

### Step 3: Verify Installation

Test the modules:

```bash
python src/text_encoder.py
python src/voxel_generator.py
python src/dataset.py
```

## 🎯 Quick Start

### Train Model

```bash
# Basic model (10-15 min CPU, 2-3 min GPU)
python train/train.py

# Diffusion model
python train/train_diffusion.py

# GNN model
python train/train_gnn.py
```

### Generate Models

```bash
# Basic model
python generate.py --text "a cube" --output cube.stl

# Diffusion (higher quality)
python generate_diffusion.py --text "a cube" --output cube.stl

# GNN (mesh generation)
python generate_gnn.py --text "a cube" --output cube.stl

# Open3D (better mesh processing)
python generate_open3d.py --text "a cube" --output cube.stl --smooth
```

## 📚 How It Works

BrickGen3D uses a simple but effective pipeline:

```
Text Input → Text Encoder → Voxel Generator → 3D Model
   ↓              ↓                ↓              ↓
"a cube"    [384D vector]    [32×32×32 grid]   🧱
```

### 1. Text Encoder

- Uses pre-trained **sentence-transformers** (MiniLM model)
- Converts text to 384-dimensional embeddings
- **Why?** It understands semantic meaning ("large cube" vs "small box")
- **Advantage**: No need to train a text encoder from scratch!

### 2. Voxel Generator

- Neural network with 3 main parts:
  1. **MLP Expansion**: 384D → 4096D (expands the embedding)
  2. **Reshape**: 4096D → 4×4×4 3D grid with 64 channels
  3. **3D Convolutions**: Upsamples to 32×32×32 final grid

- **Voxel Grid**: 3D grid where each cell is either occupied (1) or empty (0)
- **Output**: 32×32×32 = 32,768 voxels (good balance of detail vs speed)

### 3. Training

- **Dataset**: Synthetic shapes (cubes, spheres, pyramids, etc.)
- **Loss Function**: Binary Cross-Entropy (BCE)
  - Compares predicted occupancy to ground truth
  - BCE = -[y·log(p) + (1-y)·log(1-p)]
- **Optimizer**: Adam with learning rate 0.001
- **Training Time**: ~10-15 minutes for 50 epochs on CPU

### 4. Voxelization & Export

- Converts voxel grid to mesh using marching cubes
- Optional LEGO-style conversion (adds studs, gaps between bricks)
- Exports to STL, OBJ, or PLY formats

## 🎓 Training

```bash
# Basic model
python train/train.py

# Diffusion model
python train/train_diffusion.py

# GNN model
python train/train_gnn.py
```

Training saves checkpoints to `checkpoints/`. Edit training scripts to adjust epochs, batch size, etc.

### What to Expect

**After 10 epochs:**
- Model learns basic shapes
- Loss: ~0.15

**After 50 epochs:**
- Model generates recognizable shapes
- Loss: ~0.08-0.10

**After 100 epochs:**
- Better detail and accuracy
- Loss: ~0.05-0.07

## 🎨 Generating Models

```bash
# Basic
python generate.py --text "a cube" --output cube.stl

# Diffusion (higher quality)
python generate_diffusion.py --text "a cube" --output cube.stl

# GNN (mesh generation)
python generate_gnn.py --text "a cube" --output cube.stl

# Open3D (smooth mesh)
python generate_open3d.py --text "a cube" --output cube.stl --smooth
```

All scripts support `--text`, `--output`, `--checkpoint`, `--threshold` flags.

## 🏗️ Architecture Details

### Model Architecture

```
Input: Text Embedding (384D)
  ↓
FC Layer 1: 384 → 512
  ↓ (ReLU + BatchNorm + Dropout)
FC Layer 2: 512 → 1024
  ↓ (ReLU + BatchNorm + Dropout)
FC Layer 3: 1024 → 4096
  ↓ (ReLU)
Reshape: 4096 → (64, 4, 4, 4)
  ↓
ConvTranspose3D 1: 64ch, 4³ → 32ch, 8³
  ↓ (ReLU + BatchNorm)
ConvTranspose3D 2: 32ch, 8³ → 16ch, 16³
  ↓ (ReLU + BatchNorm)
ConvTranspose3D 3: 16ch, 16³ → 8ch, 32³
  ↓ (ReLU + BatchNorm)
Conv3D 4: 8ch → 1ch (occupancy)
  ↓ (Sigmoid)
Output: Voxel Grid (1, 32, 32, 32)
```

### Why This Architecture?

1. **MLP Expansion**: Text embeddings are abstract; we need to expand them into spatial information
2. **3D Convolutions**: Perfect for generating 3D structures with spatial coherence
3. **Progressive Upsampling**: Start small (4³), gradually increase detail (8³ → 16³ → 32³)
4. **BatchNorm**: Stabilizes training, allows higher learning rates
5. **Dropout**: Prevents overfitting (important for small datasets)

### Model Size

- **Parameters**: ~10 million
- **Model File**: ~40 MB
- **Memory Usage**: ~500 MB during training
- **Inference Speed**: ~0.1 seconds per model (on CPU)

## ⚠️ Limitations

### Current Limitations

1. **Simple Shapes**: Works best with basic geometric shapes (cube, sphere, pyramid)
2. **Low Resolution**: 32×32×32 voxels (1,024 total) - not high detail
3. **Synthetic Training Data**: Only trained on procedural shapes, not real objects
4. **No Color**: Generates shape only, not color/texture
5. **Size Ambiguity**: "large" vs "small" is relative to training data

### Why These Limitations?

These are **design choices** to keep the project:
- **Simple**: Easy to understand and modify
- **Fast**: Trains in minutes, not days
- **Lightweight**: Runs on CPU, no expensive GPU needed

## 🚀 Improvements

- Increase `voxel_size=64` for higher resolution
- Train longer (100+ epochs)
- Use real datasets (ShapeNet, Objaverse)
- Add color/texture generation

## 🔧 Troubleshooting

- **Checkpoint not found**: Train with `python train/train.py`
- **Out of memory**: Reduce `batch_size` in training scripts
- **Random output**: Model needs training or more epochs
- **Empty models**: Lower `--threshold` (e.g., 0.3)
- **Trimesh error**: `pip install trimesh --no-deps && pip install numpy networkx`

## 📁 Project Structure

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

## 📚 Key Concepts

- **Voxel**: 3D pixel (like Minecraft blocks)
- **Embedding**: Text → numerical representation
- **Training**: Model learns text → shape mapping
- **Generation**: Text → 3D model

See `other/` folder for detailed educational docs.


## 🤝 Contributing

This is an educational project! Feel free to:
- Experiment with different architectures
- Try real datasets
- Add new features
- Improve documentation
- Share your results!

## 🔮 Advanced Models

### Open3D
```bash
pip install open3d
python generate_open3d.py --text "a cube" --smooth
```

### Diffusion
```bash
python train/train_diffusion.py
python generate_diffusion.py --text "a cube"
```

### GNN
```bash
pip install torch-geometric
python train/train_gnn.py
python generate_gnn.py --text "a cube"
```

## 📄 License

MIT License - feel free to use this project for learning and research!

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **sentence-transformers** - Pre-trained text encoders
- **trimesh** - 3D mesh processing
- **plotly** - Interactive visualization

## 📧 Questions?

If you have questions or run into issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the code comments (they're detailed!)
3. Experiment with the parameters

---

## Citation

If you use this work, please cite:

```bibtex
@software{BrickGen3D,
  title={Generate 3D models from text descriptions using AI},
  author={Abaryan},
  year={2025},
  url={https://github.com/r-abaryan/BrickGen3D}
}
```

**Happy Building! 🧱🤖**