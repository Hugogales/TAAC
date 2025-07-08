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

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TAAC
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install environment-specific packages** (optional):
   ```bash
   # For CookingZoo
   pip install cooking_zoo
   
   # For Atari environments
   pip install "pettingzoo[atari]"
   
   # For other environments, check their specific installation instructions
   ```

4. **Test your setup**:
   ```bash
   python setup_environment.py
   ```

## Quick Start

### 1. Train on MPE Simple Spread (Default)

```bash
python AI/train_taac.py --env mpe_simple_spread
```

### 2. Train on CookingZoo with Custom Config

```bash
python AI/train_taac.py --env cooking_zoo --config configs/cooking_zoo.yaml
```

### 3. Train on BoxJump for 5000 episodes

```bash
python AI/train_taac.py --env boxjump --episodes 5000
```

### 4. Evaluate a Trained Model

```bash
python AI/train_taac.py --env cooking_zoo --eval_only --model_path files/Models/TAAC_cooking_zoo_best.pth
```

## Configuration

TAAC uses YAML configuration files for easy customization. See the `configs/` directory for examples:

- `configs/cooking_zoo.yaml` - Optimized for cooperative cooking
- `configs/boxjump.yaml` - Tuned for physics-based tasks  
- `configs/mpe_simple_spread.yaml` - Settings for coordination tasks
- `configs/mats_gym.yaml` - Configuration for traffic scenarios

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
├── AI/
│   ├── TAAC.py              # Core TAAC algorithm
│   ├── env_wrapper.py       # Environment wrapper system  
│   └── train_taac.py        # Training script
├── configs/                 # Environment configurations
│   ├── cooking_zoo.yaml
│   ├── boxjump.yaml
│   ├── mpe_simple_spread.yaml
│   └── mats_gym.yaml
├── files/Models/            # Saved models (created automatically)
├── experiments/             # Training results (created automatically)
├── requirements.txt         # Dependencies
├── setup_environment.py     # Setup verification script
└── README.md               # This file
```

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

1. **Update `env_wrapper.py`**:
   ```python
   elif env_name == 'your_env':
       from your_env import parallel_env
       return parallel_env(**kwargs)
   ```

2. **Create configuration file**:
   ```yaml
   # configs/your_env.yaml
   environment:
     name: your_env
     env_kwargs:
       # Your environment parameters
   ```

3. **Train**:
   ```bash
   python AI/train_taac.py --env your_env --config configs/your_env.yaml
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
