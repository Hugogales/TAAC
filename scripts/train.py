#!/usr/bin/env python3
"""
TAAC Training Script - Main Entry Point
Handles argument parsing and routes to appropriate training functions
"""

import multiprocessing as mp
import os
import sys

# Set spawn method for CUDA compatibility in multiprocessing
# This must be done before any CUDA operations and before importing TAAC modules
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

import yaml
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TAAC modules after setting multiprocessing method
from src.train_taac import train_taac
from src.train_taac_parallel import train_taac_parallel
from src.env_wrapper import TAACEnvironmentWrapper
from src.AI.TAAC import TAAC


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_agent(config: dict, model_path: str) -> None:
    """Run evaluation mode with a trained model"""
    print("\n=> Starting evaluation mode...")
    
    if not model_path or not os.path.exists(model_path):
        raise ValueError("Valid model path required for evaluation: {}".format(model_path))
    
    # Create environment
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    
    print("=> Creating evaluation environment: {}".format(env_name))
    env_wrapper = TAACEnvironmentWrapper(
        env_name, 
        apply_wrappers=config['environment'].get('apply_wrappers', True),
        **env_kwargs
    )
    
    # Create TAAC agent
    env_config = env_wrapper.env_info
    training_config = config.get('training', {})
    
    # Merge model configuration if available
    if 'model' in config:
        training_config.update(config['model'])
    
    print("=> Loading model: {}".format(model_path))
    taac_agent = TAAC(env_config, training_config, mode="test")
    
    if not taac_agent.load_model(model_path):
        raise RuntimeError("Failed to load model from: {}".format(model_path))
    
    print("=> Model loaded successfully!")
    
    # Run evaluation episodes
    num_eval_episodes = config.get('logging', {}).get('eval_episodes', 5)
    print("=> Running {} evaluation episodes...".format(num_eval_episodes))
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_eval_episodes):
        print("\nEvaluation Episode {}/{}".format(episode + 1, num_eval_episodes))
        
        states, _ = env_wrapper.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < 1000:
            actions, _ = taac_agent.get_actions(list(states.values()))
            states, rewards, done, _ = env_wrapper.step(actions)
            
            episode_reward += sum(rewards)
            step_count += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print("  Episode {}: Reward = {:.2f}, Length = {}".format(episode + 1, episode_reward, step_count))
    
    # Print evaluation results
    import numpy as np
    print("\n=> Evaluation Results:")
    print("  - Mean Reward: {:.2f} ± {:.2f}".format(np.mean(total_rewards), np.std(total_rewards)))
    print("  - Mean Length: {:.1f} ± {:.1f}".format(np.mean(episode_lengths), np.std(episode_lengths)))
    print("  - Best Episode: {:.2f}".format(max(total_rewards)))
    print("  - Worst Episode: {:.2f}".format(min(total_rewards)))
    
    # Clean up
    env_wrapper.close()
    print("=> Evaluation complete!")


def determine_training_mode(config: dict, args: argparse.Namespace) -> tuple:
    """
    Determine whether to use single or parallel training based on config and args.
    
    Returns:
        (use_parallel: bool, num_parallel: int)
    """
    # Check command line override first
    if args.num_parallel is not None:
        num_parallel = args.num_parallel
    else:
        # Check config file
        num_parallel = config.get('training', {}).get('num_parallel', 1)
    
    # Rendering forces single environment mode
    if args.render and num_parallel > 1:
        print("Warning: Rendering is only supported for single-environment training.")
        print("Automatically switching to single environment mode.")
        num_parallel = 1
    
    use_parallel = num_parallel > 1
    
    return use_parallel, num_parallel


def main():
    """
    Main entry point for TAAC training script.
    Handles argument parsing and routes to appropriate training mode.
    """
    parser = argparse.ArgumentParser(
        description="Train a TAAC agent on a specified environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single environment training (auto-detected from config)
  python scripts/train.py --config configs/mpe_simple_spread.yaml
  
  # Parallel training with 8 environments
  python scripts/train.py --config configs/boxjump.yaml --num_parallel 8
  
  # Training with custom settings
  python scripts/train.py --config configs/boxjump.yaml --job_name tower_builder --episodes 2000
  
  # Training with rendering (forces single env)
  python scripts/train.py --config configs/boxjump.yaml --render
  
  # Continue training from existing model
  python scripts/train.py --config configs/boxjump.yaml --load_model files/Models/boxjump/checkpoint.pth
  
  # Evaluation mode
  python scripts/train.py --config configs/boxjump.yaml --eval_only --model_path files/Models/boxjump/best_model.pth
        """
    )
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to the configuration YAML file")
    
    # Training mode arguments
    parser.add_argument("--job_name", type=str, default=None, 
                       help="Unique name for the training job (default: from config)")
    parser.add_argument("--episodes", type=int, default=None, 
                       help="Override number of training episodes from config")
    parser.add_argument("--num_parallel", type=int, default=None, 
                       help="Override number of parallel environments (1 = single, 2+ = parallel)")
    parser.add_argument("--load_model", type=str, default=None, 
                       help="Path to pre-trained model to continue training from")
    parser.add_argument("--render", action="store_true", 
                       help="Enable rendering during training (forces single environment)")
    
    # Evaluation mode arguments
    parser.add_argument("--eval_only", action="store_true", 
                       help="Run in evaluation mode instead of training")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to trained model for evaluation (required with --eval_only)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("=> Loading configuration from: {}".format(args.config))
        config = load_config(args.config)
        
        # Override config with command-line arguments
        if args.episodes:
            config.setdefault('training', {})['episodes'] = args.episodes
            print("=> Overriding episodes to: {}".format(args.episodes))
            
        if args.render:
            config.setdefault('environment', {}).setdefault('env_kwargs', {})['render_mode'] = "human"
            print("=> Enabling rendering")
            
        if args.load_model:
            config['load_model'] = args.load_model
            print("=> Will load model from: {}".format(args.load_model))
        
        if args.job_name:
            config['job_name'] = args.job_name
            print("=> Will use job name: {}".format(args.job_name))
        
        # Handle evaluation mode
        if args.eval_only:
            if not args.model_path:
                print("Error: --model_path is required when using --eval_only")
                return 1
            
            evaluate_agent(config, args.model_path)
            return 0
        
        # Determine training mode (single vs parallel)
        use_parallel, num_parallel = determine_training_mode(config, args)
        
        if use_parallel:
            print("\n=> Starting parallel training with {} environments...".format(num_parallel))
            
            # Validate parallel settings
            if num_parallel < 1:
                print("Error: Number of parallel environments must be at least 1")
                return 1
            
            # Run parallel training
            train_taac_parallel(config, num_parallel_games=num_parallel)
            
        else:
            print("\n=> Starting single environment training...")
            
            # Run single environment training
            train_taac(config)
        
        print("\n=> Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting gracefully...")
        return 0
        
    except Exception as e:
        print("\n--- An unexpected error occurred ---")
        print("Error: {}".format(e))
        print("\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 