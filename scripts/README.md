# TAAC Scripts Directory

This directory contains different execution modes for the TAAC multi-agent reinforcement learning system.

## Scripts Overview

### 🚀 `train.py` - Main Training Script
Train TAAC models using configuration files.

```bash
# Basic training
python scripts/train.py --config configs/boxjump.yaml

# Training with custom episodes and parallel environments  
python scripts/train.py --config configs/boxjump.yaml --episodes 1000 --num_parallel 8

# Continue training from existing model (set load_model in config)
python scripts/train.py --config configs/boxjump.yaml

# Evaluation only
python scripts/train.py --config configs/boxjump.yaml --eval_only --model_path files/Models/boxjump/best_model.pth
```

### 👁️ `view.py` - Model Visualization
Load and display trained models playing in their environments.

```bash
# View model using config's load_model field
python scripts/view.py --config configs/boxjump.yaml

# View specific model
python scripts/view.py --config configs/boxjump.yaml --model_path files/Models/boxjump/best_model.pth

# View for specific number of episodes
python scripts/view.py --config configs/mpe_simple_spread.yaml --episodes 3

# View with faster playback
python scripts/view.py --config configs/cooking_zoo.yaml --render_delay 0.01
```

### 📊 `profiler.py` - Performance Profiling
Profile training performance with detailed analysis and snakeviz compatibility.

```bash
# Profile training with default episodes (5)
python scripts/profiler.py --config configs/boxjump.yaml

# Profile with specific episode count
python scripts/profiler.py --config configs/mpe_simple_spread.yaml --episodes 10

# Profile and immediately view with snakeviz
python scripts/profiler.py --config configs/cooking_zoo.yaml --view

# View profile results
pip install snakeviz
snakeviz profiles/training_profile_boxjump_5ep_TIMESTAMP.prof
```

## Configuration Files

All scripts use YAML configuration files in the `configs/` directory. The available configs include:

- `configs/boxjump.yaml` - Physics-based tower building
- `configs/mpe_simple_spread.yaml` - Multi-agent particle environment 
- `configs/cooking_zoo.yaml` - Cooperative cooking tasks
- `configs/mats_gym.yaml` - Traffic coordination scenarios
- `configs/mpe_continuous.yaml` - Continuous action spaces

### New `load_model` Field

All configuration files now support an optional `load_model` field for:

1. **Continuing training** from an existing model
2. **Model visualization** with `view.py`
3. **Transfer learning** between experiments

```yaml
# Model loading (optional)
load_model: "files/Models/boxjump/best_model.pth"  # Set path to existing model
# load_model: null  # Or set to null for training from scratch
```

## Directory Structure

```
scripts/
├── README.md          # This file
├── train.py           # Main training script (replaces run_taac.py)
├── view.py            # Model visualization script
└── profiler.py        # Performance profiling script

profiles/              # Created by profile.py
└── training_profile_*.prof  # Profile files for snakeviz

configs/               # Configuration files
├── boxjump.yaml
├── mpe_simple_spread.yaml
├── cooking_zoo.yaml
├── mats_gym.yaml
└── mpe_continuous.yaml
```

## Example Workflows

### 1. Train and Visualize
```bash
# Train a model
python scripts/train.py --config configs/boxjump.yaml --episodes 100

# View the trained model
python scripts/view.py --config configs/boxjump.yaml
```

### 2. Continue Training
```bash
# First, update config file to set load_model path
# Then continue training
python scripts/train.py --config configs/boxjump.yaml --episodes 200
```

### 3. Performance Analysis
```bash
# Profile training to find bottlenecks
python scripts/profiler.py --config configs/boxjump.yaml --episodes 5 --view

# This opens snakeviz for interactive analysis
```

### 4. Model Comparison
```bash
# Train baseline model
python scripts/train.py --config configs/boxjump.yaml --episodes 100

# Set load_model in config, then train with different parameters
python scripts/train.py --config configs/boxjump.yaml --episodes 200

# Compare both models with view.py
```

## Migration from `run_taac.py`

The old `run_taac.py` has been replaced by `scripts/train.py`. The API is identical:

```bash
# Old way
python run_taac.py --config configs/boxjump.yaml

# New way  
python scripts/train.py --config configs/boxjump.yaml
```

All existing command line arguments and configuration files work exactly the same way. 