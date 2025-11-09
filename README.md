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

### Option 1: Train Your Own Model (Recommended)

```bash
# Train the model (takes ~10-15 minutes on CPU, ~2-3 minutes on GPU)
python train.py

# Generate a 3D model
python generate.py --text "a large blue cube" --output my_cube.stl
```

### Option 2: Generate Without Training (Random Output)

```bash
# This will work but produce random shapes (model is untrained)
python generate.py --text "a red pyramid" --output test.stl
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

### Basic Training

```bash
python train.py
```

This will:
- Create 5,000 training samples (synthetic shapes)
- Train for 50 epochs (~10-15 minutes on CPU)
- Save checkpoints to `checkpoints/`
- Generate training curves in `checkpoints/training_curves.png`

### Training Parameters

Edit `train.py` to customize:

```python
trainer.train(
    num_epochs=50,          # More epochs = better results (try 100)
    batch_size=64,          # Increase if you have GPU memory
    num_train_samples=5000, # More samples = more variety
    num_val_samples=500,
    save_dir='checkpoints'
)
```

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

### Basic Usage

```bash
python generate.py --text "a large red cube"
```

### Advanced Options

```bash
# Generate with LEGO style
python generate.py --text "a small pyramid" --lego_style --output pyramid.stl

# Export to OBJ format
python generate.py --text "a blue sphere" --output sphere.obj --format obj

# Don't show visualization (faster)
python generate.py --text "a cylinder" --no_show --output cylinder.stl

# Adjust voxel threshold (lower = more voxels)
python generate.py --text "a cube" --threshold 0.3 --output dense_cube.stl

# Save interactive HTML visualization
python generate.py --text "a cone" --save_html visualization.html
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--text, -t` | Text description (required) | - |
| `--checkpoint, -c` | Path to model checkpoint | `checkpoints/best_model.pth` |
| `--output, -o` | Output file path | Auto-generated |
| `--format, -f` | Export format (stl/obj/ply) | `stl` |
| `--lego_style, -l` | Use LEGO brick style | `False` |
| `--threshold` | Voxel occupancy threshold | `0.5` |
| `--no_show` | Don't show visualization | `False` |
| `--save_html` | Save HTML visualization | `None` |

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

## 🚀 Future Improvements

### Easy Improvements (You Can Try!)

1. **Increase Resolution**: Change `voxel_size=32` to `64` (slower but more detail)
2. **More Epochs**: Train for 100-200 epochs instead of 50
3. **Larger Hidden Dimensions**: Increase `hidden_dim=512` to `1024`
4. **Better Dataset**: Use real 3D models from ShapeNet or Objaverse

### Advanced Improvements

1. **Diffusion Models**: Use denoising diffusion for better quality
2. **Color Generation**: Add RGB values to each voxel
3. **Multi-View Generation**: Generate from multiple angles
4. **Image-to-3D**: Add image encoder alongside text encoder
5. **Conditional Generation**: Control size, orientation, style separately
6. **VAE/GAN Architecture**: For more diverse outputs

### Using Real Datasets

**ShapeNet** (55 categories, 51K models):
```python
# Download ShapeNet and modify dataset.py
# Convert meshes to voxels using trimesh.voxel
```

**Objaverse** (800K+ models):
```python
# Use objaverse library
import objaverse
objects = objaverse.load_objects()
```

## 🔧 Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Train the model first:
```bash
python train.py
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in `train.py`:
```python
batch_size=8  # or even 4
```

### Issue: "Generated models look random"

**Possible causes:**
1. Model not trained yet → Run `python train.py`
2. Not trained long enough → Increase epochs to 100
3. Text description too complex → Try simple shapes first

### Issue: "Trimesh import error"

**Solution**:
```bash
pip install trimesh --no-deps
pip install numpy networkx pillow
```

### Issue: "Generated models are empty"

**Solution**: Lower the threshold:
```bash
python generate.py --text "..." --threshold 0.3
```

### Issue: Slow training

**Tips:**
- Use GPU if available (automatic)
- Reduce `num_train_samples` to 2000
- Reduce `batch_size` (doesn't affect speed much)
- Use fewer epochs for testing

## 📁 Project Structure

```
BrickGen3D/
├── src/
│   ├── __init__.py
│   ├── text_encoder.py       # Text → embeddings
│   ├── voxel_generator.py    # Core model (text → voxels)
│   ├── dataset.py            # Synthetic shape dataset
│   └── voxel_utils.py        # Visualization & export
├── train.py                  # Training script
├── generate.py               # Generation script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .gitignore
├── checkpoints/              # Saved models (created during training)
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_curves.png
└── outputs/                  # Generated 3D models (created during generation)
    ├── model1.stl
    └── model2.obj
```

## 🎓 Learning Resources

### Understanding the Code

Each module has extensive comments explaining:
- **What** the code does
- **Why** we made this design choice
- **How** it fits into the bigger picture

Start here:
1. `src/text_encoder.py` - Simplest module
2. `src/dataset.py` - See how training data is created
3. `src/voxel_generator.py` - The core model
4. `train.py` - See the training loop

### Key Concepts

**Voxel**: 3D pixel (volume element)
- Like a Minecraft block
- Either occupied (1) or empty (0)

**Embedding**: Numerical representation
- Text → numbers that capture meaning
- Similar words have similar embeddings

**Binary Cross-Entropy**: Loss function
- Measures how wrong predictions are
- Good for binary classification (occupied/empty)

**Convolution**: Sliding window operation
- 2D: Used in image processing
- 3D: Used in volumetric (3D) processing

### Further Reading

- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [3D Deep Learning Survey](https://arxiv.org/abs/2004.06674)
- [Text-to-3D Methods](https://arxiv.org/abs/2303.13508)
- [Voxel-based 3D Generation](https://arxiv.org/abs/1608.04236)

## 📝 Examples

### Example Prompts

**Basic Shapes:**
```
- "a large blue cube"
- "a small red sphere"
- "a medium green pyramid"
- "a yellow cylinder"
```

**Combined (after training on combined dataset):**
```
- "a cube on top of a sphere"
- "two small pyramids"
- "a tower of blocks"
```

## 🤝 Contributing

This is an educational project! Feel free to:
- Experiment with different architectures
- Try real datasets
- Add new features
- Improve documentation
- Share your results!

## 🔮 Next Stage: Advanced Techniques

Implemented extensions in `src/`:

### 1. Open3D (Easy ⭐) - `src/open3d_utils.py`
```bash
pip install open3d
python -c "from src.open3d_utils import voxel_to_mesh; print('✓ Ready')"
```
**Features:** Better mesh processing, smoothing, simplification

### 2. Diffusion Model (Medium ⭐⭐) - `src/diffusion_model.py`
```bash
python train_diffusion.py  # Train diffusion model
```
**Features:** UNet3D architecture, 50-step denoising, higher quality

### 3. GNN (Advanced ⭐⭐⭐) - `src/gnn_model.py`
```bash
pip install torch-geometric
python train_gnn.py  # Train GNN model
```
**Features:** Direct mesh generation, graph convolutions, better topology

**All modules use the same text encoder - only generator changes!**

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

**Happy Building! 🧱🤖**

