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
from pathlib import Path
from typing import Dict, Any, Optional

from .TAAC import TAAC
from .env_wrapper import TAACEnvironmentWrapper
from .logger import extract_environment_metrics


def load_config(config_path: str) -> Dict[str, Any]:
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


def find_model_path(config: Dict[str, Any], provided_path: Optional[str] = None) -> str:
    """Find the model path from config or provided path."""
    if provided_path:
        if os.path.exists(provided_path):
            return provided_path
        else:
            raise FileNotFoundError(f"Provided model path does not exist: {provided_path}")
    
    # Check if load_model is specified in config
    if 'load_model' in config and config['load_model']:
        if os.path.exists(config['load_model']):
            return config['load_model']
    
    # Try to find a model in the default location
    env_name = config['environment']['name']
    default_paths = [
        f"files/Models/{env_name}/best_model.pth",
        f"files/Models/{env_name}/final_model.pth",
        f"files/Models/{env_name}_best.pth",
        f"files/Models/{env_name}_final.pth"
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            print(f"=> Found model at: {path}")
            return path
    
    raise FileNotFoundError(f"No model found. Tried: {default_paths}")


def load_model(model_path: str, env_wrapper: TAACEnvironmentWrapper, config: Dict[str, Any]) -> TAAC:
    """Load the trained TAAC model."""
    print(f"=> Loading model from: {model_path}")
    
    # Get environment specs
    env_config = env_wrapper.env_info
    training_config = config.get('training', {})
    
    # Merge model configuration if available
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


def play_game_ai(config: Dict[str, Any], model_path: str, episodes: int = 5, 

                render_delay: float = 0.05, interactive: bool = True) -> None:
    """Run AI visualization for specified number of episodes"""
    
    # Setup environment
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    max_steps = config['training'].get('max_steps', 1000)
    
    # Force rendering mode
    env_kwargs['render_mode'] = 'human'
    
    print(f"=> Creating environment: {env_name}")
    env_wrapper = TAACEnvironmentWrapper(env_name, **env_kwargs)
    
    # Load model
    taac_agent = load_model(model_path, env_wrapper, config)
    
    # Initialize pygame
    pygame.init()
    
    print(f"\n=> Starting AI visualization for {episodes} episodes...")
    if interactive:
        print(f"=> Press Ctrl+C to stop early, Enter to continue between episodes")
    else:
        print(f"=> Running automatically with {render_delay}s delay between steps")
    
    try:
        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            
            # Reset environment and get initial states
            states, info = env_wrapper.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            # Track environment-specific metrics
            episode_metrics = []
            
            while not done and step_count < max_steps:  # Max steps per episode
                # Get actions from the agent
                actions, _ = taac_agent.get_actions(list(states.values()))
                
                # Step the environment
                next_states, rewards, dones, infos = env_wrapper.step(actions)
                
                # Extract environment metrics for this step
                env_metrics = extract_environment_metrics(
                    env_name, list(next_states.values()), rewards, infos
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

                if all(dones.values()):
                    done = True
                    break
                
                # Control visualization speed
                time.sleep(render_delay)
            
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
            
            # Wait for user input before next episode (if interactive)
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
        # Clean up
        if hasattr(env_wrapper, 'close'):
            env_wrapper.close()
        pygame.quit()
        print(f"=> Visualization complete!")


def replay_game(log_file_path: str) -> None:
    """Replay a previously recorded game from log file"""
    print(f"=> Loading game log from: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    with open(log_file_path, "r") as file:
        log_data = json.load(file)
    
    if "states" not in log_data:
        raise ValueError("Log file does not contain game states")
    
    print("=> Replaying game...")
    # Note: This would need to be implemented based on your specific log format
    # and environment replay capabilities
    print("Game replay functionality would be implemented here") 