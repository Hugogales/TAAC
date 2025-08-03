#!/usr/bin/env python3
"""
TAAC Training Script - Config-Driven Training and Evaluation

This script is the main entry point for all TAAC training.
It automatically determines whether to use single or parallel training
based on configuration and command line arguments.

Usage:
    # Single environment training
    python scripts/train.py --config configs/boxjump.yaml
    
    # Parallel training (uses config num_parallel or override)
    python scripts/train.py --config configs/boxjump.yaml --num_parallel 8
    
    # Evaluation mode
    python scripts/train.py --config configs/boxjump.yaml --eval_only --model_path files/Models/boxjump/best_model.pth
    
    # Training with custom job name
    python scripts/train.py --config configs/mpe_simple_spread.yaml --job_name my_experiment --episodes 2000
"""

import sys
import os
import argparse
import traceback
import yaml

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import training functions
try:
    from AI.train_taac import train_taac
    from AI.train_taac_parallel import train_taac_parallel
    from AI.env_wrapper import TAACEnvironmentWrapper
    from AI.TAAC import TAAC
except ImportError as e:
    print("Error: Could not import training modules.")
    print(f"Please ensure that the AI directory and training files exist: {e}")
    sys.exit(1)


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
        raise ValueError(f"Valid model path required for evaluation: {model_path}")
    
    # Create environment
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    
    print(f"=> Creating evaluation environment: {env_name}")
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
    
    print(f"=> Loading model: {model_path}")
    taac_agent = TAAC(env_config, training_config, mode="test")
    
    if not taac_agent.load_model(model_path):
        raise RuntimeError(f"Failed to load model from: {model_path}")
    
    print(f"=> Model loaded successfully!")
    
    # Run evaluation episodes
    num_eval_episodes = config.get('logging', {}).get('eval_episodes', 5)
    print(f"=> Running {num_eval_episodes} evaluation episodes...")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_eval_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{num_eval_episodes}")
        
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
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {step_count}")
    
    # Print evaluation results
    import numpy as np
    print(f"\n=> Evaluation Results:")
    print(f"  - Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  - Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  - Best Episode: {max(total_rewards):.2f}")
    print(f"  - Worst Episode: {min(total_rewards):.2f}")
    
    # Clean up
    env_wrapper.close()
    print(f"=> Evaluation complete!")


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
        print(f"=> Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command-line arguments
        if args.episodes:
            config.setdefault('training', {})['episodes'] = args.episodes
            print(f"=> Overriding episodes to: {args.episodes}")
            
        if args.render:
            config.setdefault('environment', {}).setdefault('env_kwargs', {})['render_mode'] = "human"
            print("=> Enabling rendering")
            
        if args.load_model:
            config['load_model'] = args.load_model
            print(f"=> Will load model from: {args.load_model}")
        
        if args.job_name:
            config['job_name'] = args.job_name
            print(f"=> Will use job name: {args.job_name}")
        
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
            print(f"\n=> Starting parallel training with {num_parallel} environments...")
            
            # Validate parallel settings
            if num_parallel < 1:
                print("Error: Number of parallel environments must be at least 1")
                return 1
            
            # Run parallel training
            train_taac_parallel(config, num_parallel_games=num_parallel)
            
        else:
            print(f"\n=> Starting single environment training...")
            
            # Run single environment training
            train_taac(config)
        
        print(f"\n=> Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user. Exiting gracefully...")
        return 0
        
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 