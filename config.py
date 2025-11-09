"""
Configuration File for BrickGen3D
==================================
This file contains all configurable parameters in one place.
Edit these values to customize the model without changing the code!
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Text encoder settings
    'text_encoder_model': 'all-MiniLM-L6-v2',  # Sentence transformer model
    'text_dim': 384,                            # Embedding dimension (don't change unless you change the model)
    
    # Voxel generator settings
    'voxel_size': 32,       # Size of output grid (32 = 32×32×32 = 32,768 voxels)
                           # Options: 16 (fast, low detail), 32 (balanced), 64 (slow, high detail)
    
    'hidden_dim': 512,     # Hidden layer size
                           # Options: 256 (faster, less capacity), 512 (balanced), 1024 (slower, more capacity)
    
    # Device
    'device': 'auto',      # 'auto', 'cuda', or 'cpu'
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Dataset
    'num_train_samples': 5000,   # Number of training samples per epoch
                                 # More = more variety, longer epochs
    
    'num_val_samples': 500,      # Number of validation samples
    
    'dataset_type': 'simple',    # 'simple' or 'combined' (combined includes multi-shape scenes)
    
    # Training hyperparameters
    'num_epochs': 50,            # Number of training epochs
                                 # Recommended: 50-100 for good results
    
    'batch_size': 64,            # Batch size
                                 # Increase if you have GPU memory (32, 64)
                                 # Decrease if out of memory (8, 4)
    
    'learning_rate': 0.001,      # Initial learning rate (Adam optimizer)
    
    'weight_decay': 1e-5,        # L2 regularization strength
    
    'dropout_rate': 0.3,         # Dropout probability (0.0-0.5)
    
    # Learning rate scheduler
    'scheduler_patience': 5,     # Epochs to wait before reducing LR
    'scheduler_factor': 0.5,     # LR reduction factor
    
    # Checkpoints
    'save_dir': 'checkpoints',   # Directory to save models
    'save_every_n_epochs': 10,   # Save checkpoint every N epochs
}


# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

GENERATION_CONFIG = {
    # Default generation settings
    'threshold': 0.5,            # Voxel occupancy threshold (0.0-1.0)
                                 # Lower = more voxels, higher = fewer voxels
    
    'lego_style': False,         # Use LEGO brick style by default
    
    'output_dir': 'outputs',     # Directory for generated models
    
    'default_format': 'stl',     # Default export format ('stl', 'obj', 'ply')
    
    # Visualization
    'show_visualization': True,  # Show interactive 3D plot
    'auto_save_html': False,     # Automatically save HTML visualization
}


# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

ADVANCED_CONFIG = {
    # Data loading
    'num_workers': 0,            # DataLoader workers (0 for Windows, 2-4 for Linux/Mac)
    
    # Gradient clipping
    'grad_clip_max_norm': 1.0,   # Maximum gradient norm
    
    # Seed for reproducibility
    'random_seed': 42,           # Set to None for random behavior
    
    # Logging
    'verbose': True,             # Print detailed information
    'log_interval': 10,          # Log every N batches
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device():
    """Get the device to use for training/inference."""
    import torch
    
    if MODEL_CONFIG['device'] == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return MODEL_CONFIG['device']


def print_config():
    """Print all configuration settings."""
    print("="*60)
    print("BrickGen3D Configuration")
    print("="*60)
    
    print("\n[MODEL]")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key:25} = {value}")
    
    print("\n[TRAINING]")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key:25} = {value}")
    
    print("\n[GENERATION]")
    for key, value in GENERATION_CONFIG.items():
        print(f"  {key:25} = {value}")
    
    print("\n[ADVANCED]")
    for key, value in ADVANCED_CONFIG.items():
        print(f"  {key:25} = {value}")
    
    print("="*60)


# ============================================================================
# PRESETS
# ============================================================================

# Quick test preset (fast training for testing)
PRESET_QUICK_TEST = {
    'num_epochs': 10,
    'num_train_samples': 1000,
    'num_val_samples': 100,
    'voxel_size': 16,
    'hidden_dim': 256,
}

# High quality preset (longer training, better results)
PRESET_HIGH_QUALITY = {
    'num_epochs': 100,
    'num_train_samples': 10000,
    'num_val_samples': 1000,
    'voxel_size': 32,
    'hidden_dim': 1024,
}

# High resolution preset (detailed models, slow)
PRESET_HIGH_RES = {
    'num_epochs': 100,
    'num_train_samples': 8000,
    'num_val_samples': 800,
    'voxel_size': 64,
    'hidden_dim': 1024,
}


def apply_preset(preset_name):
    """Apply a preset configuration."""
    presets = {
        'quick_test': PRESET_QUICK_TEST,
        'high_quality': PRESET_HIGH_QUALITY,
        'high_res': PRESET_HIGH_RES,
    }
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(presets.keys())}")
        return
    
    preset = presets[preset_name]
    
    # Update configs
    for key, value in preset.items():
        if key in MODEL_CONFIG:
            MODEL_CONFIG[key] = value
        elif key in TRAINING_CONFIG:
            TRAINING_CONFIG[key] = value
    
    print(f"Applied preset: {preset_name}")
    print_config()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print_config()
    
    print("\n" + "="*60)
    print("Example: Apply Quick Test Preset")
    print("="*60)
    apply_preset('quick_test')

