#!/usr/bin/env python3
"""
TAAC Training Script - Config-Driven Training and Evaluation

Usage:
    python scripts/train.py --config configs/boxjump.yaml
    python scripts/train.py --config configs/mpe_simple_spread.yaml --eval_only
    python scripts/train.py --config configs/cooking_zoo.yaml --episodes 1000
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add AI directory to path
sys.path.append(str(Path(__file__).parent.parent / "AI"))

from train_taac import main_with_args as train_main


def load_config(config_path):
    """Load and validate configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['environment', 'training', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def setup_directories(config):
    """Create necessary directories based on config."""
    # Create model save directory
    model_path = Path(config.get('output', {}).get('model_save_path', 'files/Models/'))
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Create experiment log directory  
    log_path = Path(config.get('output', {}).get('log_save_path', 'experiments/'))
    log_path.mkdir(parents=True, exist_ok=True)
    
    return model_path, log_path


def config_to_args(config, args):
    """Convert config file to command line arguments for train_taac.py."""
    cmd_args = []
    
    # Environment
    env_name = config['environment']['name']
    cmd_args.extend(['--env', env_name])
    
    # Training parameters
    training = config.get('training', {})
    if 'episodes' in training:
        cmd_args.extend(['--episodes', str(training['episodes'])])
    if 'learning_rate' in training:
        cmd_args.extend(['--learning_rate', str(training['learning_rate'])])
    if 'gamma' in training:
        cmd_args.extend(['--gamma', str(training['gamma'])])
    
    # Logging parameters
    logging = config.get('logging', {})
    if 'log_interval' in logging:
        cmd_args.extend(['--log_interval', str(logging['log_interval'])])
    if 'save_interval' in logging:
        cmd_args.extend(['--save_interval', str(logging['save_interval'])])
    if 'eval_interval' in logging:
        cmd_args.extend(['--eval_interval', str(logging['eval_interval'])])
    
    # Output paths
    output = config.get('output', {})
    if 'model_save_path' in output:
        cmd_args.extend(['--model_save_path', output['model_save_path']])
    
    # Check for load_model in config
    if 'load_model' in config and config['load_model']:
        cmd_args.extend(['--model_path', str(config['load_model'])])
        print(f"=> Loading existing model: {config['load_model']}")
    
    # Override with command line arguments
    if args.episodes:
        # Remove existing episodes arg if present
        if '--episodes' in cmd_args:
            idx = cmd_args.index('--episodes')
            cmd_args = cmd_args[:idx] + cmd_args[idx+2:]
        cmd_args.extend(['--episodes', str(args.episodes)])
    
    if args.eval_only:
        cmd_args.append('--eval_only')
        if args.model_path:
            cmd_args.extend(['--model_path', args.model_path])
    
    if args.render:
        cmd_args.append('--render')
    
    # Handle num_parallel with proper precedence: command line > config > default
    final_num_parallel = args.num_parallel  # Command line override
    if final_num_parallel is None:
        final_num_parallel = training.get('num_parallel', 4)  # Config file or default
    cmd_args.extend(['--num_parallel', str(final_num_parallel)])
    
    return cmd_args


def main():
    parser = argparse.ArgumentParser(
        description='TAAC Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on default config (4 parallel environments)
  python scripts/train.py --config configs/mpe_simple_spread.yaml
  
  # Train with parallel environments for faster training
  python scripts/train.py --config configs/boxjump.yaml --episodes 5000 --num_parallel 8
  
  # Train with single environment (traditional mode)
  python scripts/train.py --config configs/boxjump.yaml --num_parallel 1
  
  # Evaluate trained model
  python scripts/train.py --config configs/boxjump.yaml --eval_only --model_path files/Models/boxjump/best_model.pth
  
  # Train with rendering (automatically uses single environment)
  python scripts/train.py --config configs/cooking_zoo.yaml --render --num_parallel 1
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, 
                       help='Path to YAML configuration file')
    
    # Optional overrides
    parser.add_argument('--episodes', type=int,
                       help='Override number of training episodes')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not train')
    parser.add_argument('--model_path', type=str,
                       help='Path to model for evaluation (required with --eval_only)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training/evaluation')
    parser.add_argument('--num_parallel', type=int, default=None,
                       help='Number of parallel environments for training (default: from config or 4, use 1 for single env)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_only and not args.model_path:
        parser.error("--model_path is required when using --eval_only")
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Setup directories
        model_path, log_path = setup_directories(config)
        print(f"Model save path: {model_path}")
        print(f"Log save path: {log_path}")
        
        # Show environment info
        env_name = config['environment']['name']
        print(f"Environment: {env_name}")
        
        if args.eval_only:
            print(f"Mode: Evaluation only")
            print(f"Model path: {args.model_path}")
        else:
            episodes = args.episodes or config.get('training', {}).get('episodes', 1000)
            # Get the final num_parallel value that will be used
            final_num_parallel = args.num_parallel
            if final_num_parallel is None:
                final_num_parallel = config.get('training', {}).get('num_parallel', 4)
            
            parallel_mode = "Parallel" if final_num_parallel > 1 else "Single Environment"
            print(f"Mode: {parallel_mode} Training for {episodes} episodes")
            if final_num_parallel > 1:
                print(f"Parallel environments: {final_num_parallel}")
        
        # Convert config to command line arguments
        cmd_args = config_to_args(config, args)
        print(f"Running: train_taac.py {' '.join(cmd_args)}")
        
        # Create mock args object for train_taac
        import argparse as ap
        train_parser = ap.ArgumentParser()
        train_parser.add_argument('--env', default='mpe_simple_spread')
        train_parser.add_argument('--episodes', type=int, default=1000)
        train_parser.add_argument('--learning_rate', type=float, default=3e-4)
        train_parser.add_argument('--gamma', type=float, default=0.99)
        train_parser.add_argument('--log_interval', type=int, default=10)
        train_parser.add_argument('--save_interval', type=int, default=100)
        train_parser.add_argument('--eval_interval', type=int, default=50)
        train_parser.add_argument('--model_save_path', default='files/Models/')
        train_parser.add_argument('--eval_only', action='store_true')
        train_parser.add_argument('--model_path', default=None)
        train_parser.add_argument('--render', action='store_true')
        train_parser.add_argument('--num_parallel', type=int, default=4)
        
        train_args = train_parser.parse_args(cmd_args)
        
        # Add config to args for train_taac to use
        train_args.config = config
        
        # Run training/evaluation
        train_main(train_args)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 