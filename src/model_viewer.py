#!/usr/bin/env python3
"""
TAAC Model Viewer - Load and Display Trained Models
Core viewing functionality without argument parsing
"""

import argparse
import yaml
import os
import sys
import time
import torch
import pygame
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .AI.TAAC import TAAC
from .env_wrapper import TAACEnvironmentWrapper
from .logger import extract_environment_metrics
from tqdm import tqdm


def normalize_entropy(entropy_dict: Dict[str, float], num_actions: int) -> float:
    """Normalize entropy to 0-1 scale"""
    if not entropy_dict:
        return None
    
    avg_entropy = np.mean(list(entropy_dict.values()))
    
    if num_actions is None or num_actions <= 1:
        return None
    max_entropy = np.log(num_actions)
    min_entropy = 0.0
    
    if max_entropy == min_entropy:
        return 0.5
    
    normalized = (avg_entropy - min_entropy) / (max_entropy - min_entropy)
    return max(0.0, min(1.0, normalized))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def find_model_path(base_path: str, env_name: str) -> str:
    """Find the best model for the given environment."""
    
    # Try exact path first
    if os.path.exists(base_path):
        return base_path
    
    # Look for model in standard locations
    search_paths = [
        f"files/Models/{env_name}/best_model.pth",
        f"files/Models/{env_name}_best.pth",
        f"Models/{env_name}/best_model.pth",
        f"Models/{env_name}_best.pth"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find model for {env_name}. Tried: {search_paths}")


def load_model(model_path: str, env_wrapper: TAACEnvironmentWrapper, config: Dict[str, Any]) -> TAAC:
    """Load a trained TAAC model."""
    
    print(f"=> Loading model from: {model_path}")
    for i in tqdm(range(1), desc=f"Loading model {model_path}"):
        pass
    
    # Extract environment configuration
    env_config = {
        'num_agents': env_wrapper.num_agents,
        'state_size': env_wrapper.state_size,
        'action_size': env_wrapper.action_size,
        'action_space_type': env_wrapper.action_space_type
    }
    
    # Extract training configuration
    training_config = config.get('training', {}).copy()
    if 'model' in config:
        training_config.update(config['model'])
    
    print(f"Environment specs:")
    print(f"  - Agents: {env_config['num_agents']}")
    print(f"  - State size: {env_config['state_size']}")
    print(f"  - Action size: {env_config['action_size']}")
    print(f"  - Action type: discrete")
    
    # Create TAAC agent
    taac_agent = TAAC(env_config, training_config, mode="test")
    
    # Load the saved model
    if not taac_agent.load_model(model_path):
        raise RuntimeError("Model loading failed. Please check the model path.")
        
    print(f"=> Model loaded successfully!")
    return taac_agent


def play_game_ai(config: Dict[str, Any], model_path: str, episodes: int = 2, 
                render_delay: float = 0.05, interactive: bool = True) -> None:
    """Run AI visualization for specified number of episodes"""
    
    # Setup environment
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    max_steps = config['training'].get('max_steps', 1000)
    
    # Force rendering mode
    env_kwargs['render_mode'] = 'human'
    
    print(f"=> Creating environment: {env_name}")
    
    try:
        # Create environment wrapper
        env_wrapper = TAACEnvironmentWrapper(env_name, **env_kwargs)
        
        # Load model
        taac_agent = load_model(model_path, env_wrapper, config)
        
        print(f"\n=> Starting AI visualization for {episodes} episodes...")
        if interactive:
            print(f"=> Press Ctrl+C to stop early, Enter to continue between episodes")
        else:
            print(f"=> Running automatically with {render_delay}s delay between steps")
        
        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            
            # Initialize pygame for this episode
            pygame_initialized = False
            try:
                pygame.init()
                pygame.display.set_caption(f"TAAC Model Viewer - {env_name} - Episode {episode + 1}")
                pygame_initialized = True
                print(f"=> Pygame window opened for episode {episode + 1}")
            except Exception as e:
                print(f"Warning: Could not initialize pygame for episode {episode + 1}: {e}")
                print(f"=> Skipping rendering for this episode")
                # Temporarily disable rendering for this episode
                original_render_mode = env_kwargs.get('render_mode')
                env_kwargs['render_mode'] = None
            
            try:
                # Reset environment and get initial states
                states, info = env_wrapper.reset()
                episode_reward = 0
                step_count = 0
                done = False
                
                # Track environment-specific metrics
                episode_metrics = []
                
                while not done and step_count < max_steps:  # Max steps per episode
                    # Get actions from the agent
                    actions, _log_probs, _entropies = taac_agent.get_actions(states)
                    
                    # Step the environment
                    next_states, rewards, dones, infos = env_wrapper.step(actions)
                    
                    # Extract environment metrics for this step
                    env_metrics = extract_environment_metrics(
                        env_name, next_states, rewards, infos
                    )
                    if env_metrics:
                        episode_metrics.append(env_metrics)
                    
                    # Update states and rewards
                    states = next_states
                    step_reward = sum(rewards)
                    episode_reward += step_reward
                    step_count += 1
                    
                    # Print step info with environment-specific details
                    step_info = f"  Step {step_count}: Reward = {step_reward:.2f}"
                    
                    # Add environment-specific information
                    if env_name == 'boxjump' and env_metrics and 'max_height' in env_metrics:
                        step_info += f", Height: {env_metrics['max_height']:.2f}"
                    elif env_name == 'mpe_simple_spread' and env_metrics and 'collision_count' in env_metrics:
                        step_info += f", Collisions: {env_metrics['collision_count']}"
                    elif env_name == 'cooking_zoo' and env_metrics and 'dishes_completed' in env_metrics:
                        step_info += f", Dishes: {env_metrics['dishes_completed']}"
                    elif env_name == 'mats_gym' and env_metrics and 'traffic_flow' in env_metrics:
                        step_info += f", Traffic: {env_metrics['traffic_flow']:.2f}"
                    
                    print(step_info)

                    if dones:
                        done = True
                        break
                    
                    # Control visualization speed
                    #time.sleep(render_delay)
                
                # Episode summary
                print(f"--- Episode {episode + 1} finished ---")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Episode length: {step_count} steps")
                
                # Environment-specific episode summary
                if episode_metrics:
                    if env_name == 'boxjump':
                        max_heights = [m.get('max_height', 0) for m in episode_metrics if 'max_height' in m]
                        if max_heights:
                            print(f"Max height achieved: {max(max_heights):.2f}")
                            
                    elif env_name == 'mpe_simple_spread':
                        total_collisions = sum(m.get('collision_count', 0) for m in episode_metrics)
                        print(f"Total collisions: {total_collisions}")
                        
                    elif env_name == 'cooking_zoo':
                        total_dishes = sum(m.get('dishes_completed', 0) for m in episode_metrics)
                        print(f"Total dishes completed: {total_dishes}")
                        
                    elif env_name == 'mats_gym':
                        avg_traffic = sum(m.get('traffic_flow', 0) for m in episode_metrics) / len(episode_metrics)
                        print(f"Average traffic flow: {avg_traffic:.2f}")
                
            except Exception as e:
                print(f"=> Error during episode {episode + 1}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Clean up pygame for this episode
                if pygame_initialized:
                    try:
                        pygame.display.quit()
                        pygame.quit()
                        print(f"=> Pygame window closed for episode {episode + 1}")
                    except Exception as e:
                        print(f"Warning: Error closing pygame for episode {episode + 1}: {e}")
                
                # Restore original render mode if it was temporarily disabled
                if 'original_render_mode' in locals():
                    env_kwargs['render_mode'] = original_render_mode
                    del original_render_mode
            
            # Wait for user input before next episode (if interactive and not last episode)
            if interactive and episode < episodes - 1:
                try:
                    input(f"  Press Enter to continue to episode {episode + 2}, or Ctrl+C to exit...")
                except KeyboardInterrupt:
                    print(f"\n=> Visualization stopped by user after episode {episode + 1}")
                    break
    
    except KeyboardInterrupt:
        print(f"\n=> Visualization interrupted by user")
    
    except Exception as e:
        print(f"\n=> Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Comprehensive cleanup
        try:
            # Close environment first
            if 'env_wrapper' in locals() and hasattr(env_wrapper, 'close'):
                env_wrapper.close()
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")
        
        print(f"=> Visualization complete!")


def replay_game(log_file_path: str) -> None:
    """Replay a previously recorded game from log file"""
    print(f"=> Loading game log from: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    # Initialize pygame with error handling for future replay functionality
    pygame_initialized = False
    try:
        pygame.init()
        pygame.display.set_caption("TAAC Game Replay")
        pygame_initialized = True
        print(f"=> Pygame initialized for replay")
    except Exception as e:
        print(f"Warning: Could not initialize pygame for replay: {e}")
    
    try:
        with open(log_file_path, "r") as file:
            log_data = json.load(file)
        
        if "states" not in log_data:
            raise ValueError("Log file does not contain game states")
        
        print("=> Replaying game...")
        # Note: This would need to be implemented based on your specific log format
        # and environment replay capabilities
        print("Game replay functionality would be implemented here")
        
        # Future implementation would go here:
        # - Load environment configuration from log
        # - Create environment
        # - Step through recorded states/actions
        # - Render each frame with proper timing
        
    except Exception as e:
        print(f"=> Error during replay: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up pygame if it was initialized
        if pygame_initialized:
            try:
                pygame.display.quit()
                pygame.quit()
                print(f"=> Pygame cleaned up after replay")
            except Exception as e:
                print(f"Warning: Error cleaning up pygame after replay: {e}")
        
        print(f"=> Replay complete!")