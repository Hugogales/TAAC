# TAAC - Environment-Agnostic Multi-Agent Reinforcement Learning

TAAC (previously HUGO - "Hierarchical Unified Generalized Optimization") is a multi-agent reinforcement learning algorithm based on PPO with attention mechanisms, designed to work across various PettingZoo environments.

## Features

- **Environment-Agnostic Design**: Works with any PettingZoo parallel environment
- **Attention-Based Architecture**: Multi-head attention for agent coordination  
- **Discrete Action Spaces**: Optimized for discrete action environments
- **Dynamic Network Sizing**: Automatically adapts to environment specifications
- **Centralized Training**: Uses centralized training with decentralized execution
- **Environment Wrappers**: Built-in support for CookingZoo, BoxJump, MATS Gym, and MPE
- **Configurable Training**: YAML-based configuration system
- **Comprehensive Logging**: Training curves, model checkpoints, and evaluation metrics

## Supported Environments

- **CookingZoo**: Cooperative cooking tasks
- **BoxJump**: Physics-based coordination
- **MATS Gym**: Multi-agent traffic scenarios  
- **MPE (Multi-agent Particle Environment)**: Simple coordination tasks
- **PettingZoo Atari**: Multi-agent Atari games
- **Custom environments**: Easy to add new PettingZoo environments

## Quick Setup

1. **Clone and setup virtual environment**:
   ```bash
   git clone <repository-url>
   cd TAAC
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment-specific setup**:
   ```bash
   # Install all PettingZoo environments
   pip install 'pettingzoo[all]'
   
   # Or install specific environments only
   pip install 'pettingzoo[atari]'  # Atari games
   pip install 'pettingzoo[mpe]'    # Multi-agent particle environments
   ```

4. **GPU/CUDA Setup (for parallel training)**:
   ```bash
   # Ensure PyTorch CUDA is installed
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Test CUDA multiprocessing compatibility
   python test_cuda_multiprocessing.py
   ```

5. **BoxJump environment** (if using BoxJump):
   ```bash
   # Install Box2D via conda (avoids Windows compilation issues)
   conda install -c conda-forge box2d-py
   
   # Install other dependencies
   pip install swig pygame numpy
   
   # Create environments directory if it doesn't exist
   mkdir -p environments
   cd environments
   
   # Clone BoxJump repository
   git clone https://github.com/zzbuzzard/boxjump
   cd boxjump && pip install -e . && cd ../..
   
   # Test: python -c "from environments.boxjump.box_env import BoxJumpEnvironment; print('Success!')"
   ```

6. **Quick test**:
   ```bash
   python scripts/train.py --config configs/mpe_simple_spread.yaml --episodes 10
   ```

## CUDA Multiprocessing for Supercomputers

TAAC automatically handles CUDA multiprocessing compatibility for supercomputer and multi-GPU environments:

- **Automatic Configuration**: Uses 'spawn' method instead of 'fork' for CUDA safety
- **Multi-GPU Support**: Distributes workers across available GPUs  
- **SLURM Compatibility**: Works with job schedulers and resource managers
- **Memory Efficiency**: Prevents CUDA context conflicts between processes

**Testing CUDA Setup:**
```bash
# Run the test script to verify your setup
python test_cuda_multiprocessing.py

# If successful, run parallel training
python scripts/train.py --config configs/boxjump.yaml --num_parallel 8
```

**Common Issues:**
- If you see "Cannot re-initialize CUDA in forked subprocess", the spawn method should fix this automatically
- For older PyTorch versions, you may need to manually set `CUDA_VISIBLE_DEVICES`
- On some clusters, you may need to load CUDA modules before running

## Quick Start

### 1. Train with Default Config (MPE Simple Spread)

```bash
python scripts/train.py --config configs/mpe_simple_spread.yaml
```

### 2. Train on BoxJump (Physics-Based Cooperation)

```bash
python scripts/train.py --config configs/boxjump.yaml
```

### 3. Train on CookingZoo Environment

```bash
python scripts/train.py --config configs/cooking_zoo.yaml
```

### 4. Train with Custom Episode Count

```bash
python scripts/train.py --config configs/cooking_zoo.yaml --episodes 5000
```

### 5. Evaluate a Trained Model

```bash
python scripts/train.py --config configs/mpe_simple_spread.yaml --eval_only --model_path files/Models/mpe_simple_spread/best_model.pth
```

### 6. Train with Live Rendering

```bash
python scripts/train.py --config configs/boxjump.yaml --render
```

## BoxJump Setup & Usage

BoxJump is a cooperative tower-building environment where agents (boxes) work together to build the tallest possible tower using physics simulation.

### BoxJump Installation

```bash
# 1. Install Box2D physics engine (use conda for pre-compiled binary)
conda install -c conda-forge box2d-py

# 2. Install other dependencies
pip install swig pygame numpy

# 3. Clone and install BoxJump in environments directory
mkdir -p environments
cd environments
git clone https://github.com/zzbuzzard/boxjump
cd boxjump
pip install -e .
cd ../..

# 4. Test installation
python -c "from environments.boxjump.box_env import BoxJumpEnvironment; print('BoxJump installed successfully!')"
```

**Important**: Use `conda` for `box2d-py` on Windows - pip compilation often fails. If you don't have conda, you can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

**Note**: We install BoxJump outside your TAAC repository to avoid committing someone else's entire codebase to your project. The `-e` flag installs it in "editable" mode so Python can still find it from anywhere.

#### Troubleshooting BoxJump Installation

If you encounter compilation errors with `box2d-py`, use this **proven solution**:

```bash
# ❌ If pip fails with: "error: command 'swig.exe' failed"
# ✅ Use conda instead:
conda install -c conda-forge box2d-py

# Then install BoxJump
cd boxjump
pip install -e .

# Test it works:
python -c "import Box2D; print('Box2D working!')"
python -c "from boxjump.box_env import BoxJumpEnvironment; print('BoxJump ready!')"
```

**Why this works**: Conda provides pre-compiled binaries for Box2D, avoiding Windows compilation issues with SWIG and Visual Studio dependencies.

### Running BoxJump

```bash
# Train with default settings (4 agents, cooperative tower building)
python scripts/train.py --config configs/boxjump.yaml

# Train with visualization (watch the boxes build towers!)
python scripts/train.py --config configs/boxjump.yaml --render

# Train for longer with more episodes
python scripts/train.py --config configs/boxjump.yaml --episodes 5000

# Evaluate a trained BoxJump model
python scripts/train.py --config configs/boxjump.yaml --eval_only --model_path files/Models/boxjump/best_model.pth

# Use parallel environment execution for faster training (8 processes)
python scripts/parallel_env_example.py --config configs/boxjump.yaml --num_processes 8
```

### BoxJump Configuration Options

Edit `configs/boxjump.yaml` to customize:

- **`num_boxes`**: Number of agents (2-16). Start with 4, increase for harder coordination
- **`fixed_rotation`**: `true` = easier (no rotation), `false` = harder (full physics)
- **`render_mode`**: `null` for training, `"human"`