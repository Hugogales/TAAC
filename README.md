# TAAC - Environment-Agnostic Multi-Agent Reinforcement Learning

TAAC (previously HUGO - "Hierarchical Unified Generalized Optimization") is a multi-agent reinforcement learning algorithm based on PPO with attention mechanisms, designed to work across various PettingZoo environments.

## Features

- **Environment-Agnostic Design**: Works with any PettingZoo parallel environment
- **Attention-Based Architecture**: Multi-head attention for agent coordination  
- **Supports Both Action Types**: Discrete and continuous action spaces
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

3. **Install environment-specific packages** (optional):
   ```bash
   # Install all PettingZoo environments
   pip install 'pettingzoo[all]'
   
   # Or install specific environments only
   pip install 'pettingzoo[atari]'  # Atari games
   pip install 'pettingzoo[mpe]'    # Multi-agent particle environments
   ```

4. **Install BoxJump environment** (if using BoxJump):
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

5. **Quick test**:
   ```bash
   python scripts/train.py --config configs/mpe_simple_spread.yaml --episodes 10
   ```

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
- **`render_mode`**: `null` for training, `"human"` for PyGame visualization
- **`max_cycles`**: Episode length (default: 500 steps)

### BoxJump Environment Details

- **Objective**: Build the tallest tower by stacking boxes cooperatively
- **Actions**: 4 discrete actions per agent: [do nothing, move left, move right, jump]  
- **Observations**: 13D vector per agent (position, velocity, contact info, etc.)
- **Rewards**: Shared reward when new maximum tower height is achieved
- **Physics**: Box2D simulation with realistic stacking and collision

### Difficulty Progression

```yaml
# Easy - 2 agents, no rotation
num_boxes: 2
fixed_rotation: true

# Medium - 4 agents, no rotation  
num_boxes: 4
fixed_rotation: true

# Hard - 8+ agents with full physics
num_boxes: 8
fixed_rotation: false
```

## Config-Driven Workflow

TAAC uses a simple config-driven approach: **One script (`run_taac.py`) + YAML configs** handle everything.

### Available Configurations

- `configs/mpe_simple_spread.yaml` - Multi-agent coordination (default)
- `configs/boxjump.yaml` - Physics-based environments (placeholder for custom environment)
- `configs/cooking_zoo.yaml` - Cooperative cooking tasks  
- `configs/mats_gym.yaml` - Traffic scenarios

### Config File Structure

All settings are controlled through YAML files:

### Configuration Structure

```yaml
environment:
  name: environment_name
  env_kwargs:
    # Environment-specific parameters
  apply_wrappers: true

training:
  episodes: 1000
  learning_rate: 3e-4
  gamma: 0.99
  # ... other hyperparameters

logging:
  log_interval: 10
  save_interval: 100
  eval_interval: 50

model:
  num_heads: 4
  embedding_dim: 256
  hidden_size: 526
```

## Project Structure

```
TAAC/
├── scripts/                  # 🚀 Main scripts (START HERE)
│   ├── train.py              # Training script (formerly run_taac.py)
│   ├── view.py               # Model visualization/rendering
│   ├── profiler.py           # Performance profiling
│   └── parallel_env_example.py # True parallel environment execution
├── AI/
│   ├── TAAC.py              # Core TAAC algorithm
│   ├── env_wrapper.py       # Environment wrapper system  
│   └── train_taac.py        # Training script (called by runner)
├── configs/                 # 📋 Configuration files
│   ├── mpe_simple_spread.yaml
│   ├── boxjump.yaml         # Template for custom environments
│   ├── cooking_zoo.yaml
│   └── mats_gym.yaml
├── environments/            # 🌍 Environment repositories
│   ├── boxjump/             # BoxJump environment
│   ├── cooking_zoo/         # CookingZoo environment
│   └── mats_gym/            # MATS Gym environment
├── files/                   # 💾 Model storage (organized by environment)
│   └── Models/
│       ├── boxjump/
│       ├── mpe_simple_spread/
│       ├── cooking_zoo/
│       └── mats_gym/
├── experiments/             # 📊 Training logs (organized by environment)
│   ├── boxjump/
│   ├── mpe_simple_spread/
│   ├── cooking_zoo/
│   └── mats_gym/
├── .venv/                   # Virtual environment (created by you)
├── .gitignore              # Git ignore rules
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Organization Benefits

- **Environment Isolation**: Each environment has its own model and log directories
- **Easy Management**: Models from different games don't mix together
- **Clean Structure**: Clear separation makes finding specific experiments simple
- **Scalable**: Easy to add new environments without cluttering

## Algorithm Details

### TAAC Architecture

TAAC uses a centralized training, decentralized execution approach with:

1. **Attention-Based Actor**: Multi-head self-attention over agent states
2. **Attention-Based Critic**: Centralized value function with attention
3. **Similarity Loss**: Encourages or discourages agent coordination
4. **Multi-Agent Baseline**: Counterfactual baseline for variance reduction

### Key Components

- **AttentionActorCriticNetwork**: Neural network with multi-head attention
- **Memory**: Experience replay for each agent
- **TAACEnvironmentWrapper**: Standardizes different environments
- **Dynamic Architecture**: Adapts network size based on environment specs

## Environment Integration

### Adding New Environments

To add a new PettingZoo environment:

1. **Install the environment in the environments directory**:
   ```bash
   mkdir -p environments
   cd environments
   git clone https://github.com/your-org/your_env
   cd your_env
   pip install -e .
   cd ../..
   ```

2. **Update `env_wrapper.py`**:
   ```python
   elif env_name == 'your_env':
       try:
           from environments.your_env import parallel_env
       except ImportError:
           from your_env import parallel_env
       return parallel_env(**kwargs)
   ```

3. **Create configuration file**:
   ```yaml
   # configs/your_env.yaml
   environment:
     name: your_env
     env_kwargs:
       # Your environment parameters
   ```

4. **Train**:
   ```bash
   python scripts/train.py --config configs/your_env.yaml
   ```

### Centralized vs. PettingZoo Parallel API

TAAC's centralized training approach is compatible with PettingZoo's parallel API:

- **Environment Interface**: Uses parallel API for efficiency
- **Action Collection**: Gathers actions from all agents simultaneously  
- **Centralized Updates**: Updates all agents together using shared experiences
- **No Interference**: The centralized approach enhances rather than conflicts with parallel execution

## Training Tips

### Hyperparameter Guidelines

- **Cooperative Tasks** (CookingZoo): Higher similarity loss coefficient (0.2)
- **Physics Tasks** (BoxJump): Lower learning rate (2e-4), higher gamma (0.995)
- **Short Episodes** (MPE): Higher learning rate (1e-3), fewer K epochs (4)
- **Complex Tasks**: Larger network architecture (more heads, larger embedding)

### Common Issues

1. **Environment Import Errors**: Check environment installation
2. **CUDA Out of Memory**: Reduce batch size or network size
3. **Slow Training**: Enable GPU, reduce eval frequency
4. **Poor Performance**: Adjust similarity loss coefficient for task type

## Results and Evaluation

Training results are automatically saved to `experiments/` with:

- **Training curves**: Episode rewards, lengths, evaluation scores
- **Model checkpoints**: Best model, periodic saves, final model
- **Statistics**: JSON file with detailed training statistics
- **Configuration**: Copy of config used for reproducibility

## Contributing

When adding new environments or features:

1. Test with `setup_environment.py`
2. Create appropriate configuration files
3. Update documentation
4. Ensure compatibility with existing environments

## Citation

If you use TAAC in your research, please cite:

```bibtex
@misc{taac2024,
  title={TAAC: Environment-Agnostic Multi-Agent Reinforcement Learning},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
