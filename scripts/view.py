#!/usr/bin/env python3
"""
TAAC Model Viewer - Load and Display Trained Models

Usage:
    python scripts/view.py --config configs/boxjump.yaml
    python scripts/view.py --config configs/mpe_simple_spread.yaml --model_path files/Models/mpe/best_model.pth
    python scripts/view.py --config configs/boxjump.yaml --episodes 5
"""

import argparse
import yaml
import os
import sys
import time
import torch
from pathlib import Path

# Add AI directory to path
sys.path.append(str(Path(__file__).parent.parent / "AI"))

from TAAC import TAAC
from env_wrapper import TAACEnvironmentWrapper


def load_config(config_path):
    """Load and validate configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['environment', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def find_model_path(config, provided_path):
    """Find the model path from config or provided path."""
    if provided_path:
        return provided_path
    
    # Check if load_model is specified in config
    if 'load_model' in config and config['load_model']:
        return config['load_model']
    
    # Try to find a model in the default location
    env_name = config['environment']['name']
    default_paths = [
        f"files/Models/{env_name}/best_model.pth",
        f"files/Models/{env_name}/final_model.pth",
        f"files/Models/TAAC_{env_name}_final.pth",
        f"files/Models/{env_name}_taac_final.pth"
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            print(f"=> Found model at: {path}")
            return path
    
    raise FileNotFoundError(f"No model found. Tried: {default_paths}")


def load_model(model_path, env_wrapper):
    """Load the trained TAAC model."""
    print(f"=> Loading model from: {model_path}")
    
    # Get environment specs
    state_size = env_wrapper.state_size
    action_size = env_wrapper.action_size
    action_type = env_wrapper.action_type
    num_agents = env_wrapper.num_agents
    
    print(f"Environment specs:")
    print(f"  - Agents: {num_agents}")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Action type: {action_type}")
    
    # Create TAAC agent
    taac_agent = TAAC(
        state_size=state_size,
        action_size=action_size,
        action_type=action_type,
        num_agents=num_agents
    )
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'actor_state_dict' in checkpoint:
        taac_agent.actor_critic.load_state_dict(checkpoint['actor_state_dict'])
    else:
        # Try to load direct state dict
        taac_agent.actor_critic.load_state_dict(checkpoint)
    
    print(f"=> Model loaded successfully!")
    return taac_agent


def run_visualization(config, model_path, episodes=5, render_delay=0.05):
    """Run the visualization with the trained model."""
    # Setup environment
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    
    # Force rendering mode
    env_kwargs['render_mode'] = 'human'
    
    print(f"=> Creating environment: {env_name}")
    env_wrapper = TAACEnvironmentWrapper(env_name, **env_kwargs)
    
    # Load model
    taac_agent = load_model(model_path, env_wrapper)
    
    print(f"\n=> Starting visualization for {episodes} episodes...")
    print(f"=> Press Ctrl+C to stop early")
    
    try:
        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            
            # Reset environment
            states = env_wrapper.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 1000:  # Max steps per episode
                # Get actions from trained agent
                actions, _ = taac_agent.get_actions(states)
                
                # Step environment
                next_states, rewards, done, info = env_wrapper.step(actions)
                
                # Accumulate rewards
                total_reward = sum(rewards.values()) if isinstance(rewards, dict) else sum(rewards)
                episode_reward += total_reward
                
                # Extract height info for BoxJump
                height_info = ""
                if env_name == 'boxjump' and hasattr(env_wrapper, 'get_current_height'):
                    try:
                        current_height = env_wrapper.get_current_height()
                        if current_height is not None:
                            height_info = f", Height: {current_height:.2f}"
                    except:
                        pass
                
                # Print step info periodically
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward: {total_reward:.3f}{height_info}")
                
                states = next_states
                step_count += 1
                
                # Control visualization speed
                time.sleep(render_delay)
            
            print(f"  Episode complete: Total Reward: {episode_reward:.2f}, Steps: {step_count}")
            print(f"  Press Enter to continue to next episode, or Ctrl+C to exit...")
            
            # Wait for user input before next episode
            try:
                input()
            except KeyboardInterrupt:
                print(f"\n=> Visualization stopped by user")
                break
    
    except KeyboardInterrupt:
        print(f"\n=> Visualization interrupted by user")
    
    except Exception as e:
        print(f"\n=> Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if hasattr(env_wrapper, 'close'):
            env_wrapper.close()
        print(f"=> Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='TAAC Model Viewer - Display Trained Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View model using config's load_model field
  python scripts/view.py --config configs/boxjump.yaml
  
  # View specific model
  python scripts/view.py --config configs/boxjump.yaml --model_path files/Models/boxjump/best_model.pth
  
  # View for specific number of episodes
  python scripts/view.py --config configs/mpe_simple_spread.yaml --episodes 3
  
  # View with faster playback
  python scripts/view.py --config configs/cooking_zoo.yaml --render_delay 0.01
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, 
                       help='Path to YAML configuration file')
    
    # Optional arguments
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model file (overrides config load_model)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to display (default: 5)')
    parser.add_argument('--render_delay', type=float, default=0.05,
                       help='Delay between steps in seconds (default: 0.05)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"=> Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Find model path
        model_path = find_model_path(config, args.model_path)
        
        # Run visualization
        run_visualization(config, model_path, args.episodes, args.render_delay)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 