#!/usr/bin/env python3
"""
Parallel Environment Training script for TAAC algorithm
Adapted from the provided parallel training structure with multiprocessing
"""

import yaml
import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import multiprocessing as mp_base
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import json
import pygame
from tqdm import tqdm
import math
from functools import partial

from .TAAC import TAAC
from .env_wrapper import TAACEnvironmentWrapper, create_env_config
from .logger import TAACLogger, extract_environment_metrics, format_time


# --- Constants ---
BASE_SAVE_DIR = "files"
BASE_LOG_DIR = "files/experiments"


def setup_paths(env_name: str, job_name: Optional[str] = None) -> Tuple[str, str]:
    """Create unique directories for saving models and logs."""
    if job_name is None:
        job_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    save_dir = os.path.join(BASE_SAVE_DIR, "Models", env_name, job_name)
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join(BASE_LOG_DIR, env_name, job_name)
    os.makedirs(log_dir, exist_ok=True)
    
    return save_dir, log_dir


def save_config(config: Dict[str, Any], log_dir: str):
    """Save the configuration to a YAML file in the specified directory."""
    config_path = os.path.join(log_dir, 'config.yaml')
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"-> Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def run_single_game_parallel(args):
    """Run a single game in parallel environment (for multiprocessing)"""
    if len(args) == 5:
        (train_model, gpu_id, env_name, env_kwargs, max_steps) = args
    else:
        (train_model, env_name, env_kwargs, max_steps) = args
        gpu_id = None
    
    # Set device for this process
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
    train_model.assign_device(device) if hasattr(train_model, 'assign_device') else None
    
    # Create environment for this process
    env_wrapper = TAACEnvironmentWrapper(env_name, **env_kwargs)
    
    # Reset environment
    states, info = env_wrapper.reset()
    train_model.memory_prep(env_wrapper.num_agents)
    
    episode_reward = 0
    episode_entropies = []
    step_count = 0
    done = False
    
    # Track environment-specific metrics
    all_states = []
    all_rewards = []
    all_infos = []
    
    while not done and step_count < max_steps:
        # Get actions from training model
        train_actions, train_log_probs, train_entropies = train_model.get_actions(states)
        
        # Collect entropies for logging (every 10 steps to reduce noise)
        if step_count % 10 == 0:
            episode_entropies.append(train_entropies)
        
        # Step environment
        next_states, rewards, done, env_info = env_wrapper.step(train_actions)
        
        # Store rewards and states for metrics (experience storage happens in get_actions)
        train_model.store_rewards(rewards, done)
        all_states.append(next_states)
        all_rewards.append(rewards)
        all_infos.append(env_info)
        
        # Update state and reward tracking
        states = next_states
        episode_reward += sum(rewards)
        step_count += 1
        
        if done:
            break
    
    # Calculate normalized entropy for this episode
    normalized_entropy = None
    if episode_entropies:
        avg_entropy_dict = {}
        for agent_key in episode_entropies[0].keys():
            avg_entropy_dict[agent_key] = np.mean([ent[agent_key] for ent in episode_entropies])
        
        normalized_entropy = normalize_entropy(
            avg_entropy_dict,
            env_wrapper.action_size
        )
    
    # Extract environment-specific metrics
    final_states = all_states[-1] if all_states else []
    final_rewards = all_rewards[-1] if all_rewards else {}
    final_info = all_infos[-1] if all_infos else {}
    
    env_metrics = extract_environment_metrics(env_name, final_states, final_rewards, final_info, all_states_history=all_states)
    
    # Close environment
    env_wrapper.close()
    
    # Get model memories for combining later
    memories = train_model.memories if hasattr(train_model, 'memories') else []
    
    return (episode_reward, normalized_entropy, env_metrics), memories

def train_taac_parallel(config: Dict[str, Any], num_parallel_games: int = 4) -> TAAC:
    """
    Main training loop for TAAC algorithm (parallel environments)
    """
    # Setup paths and logging
    env_name = config['environment']['name']
    save_dir, log_dir = setup_paths(env_name, config['job_name'])
    actual_job_name = os.path.basename(save_dir)
    
    # Save configuration
    save_config(config, log_dir)
    
    # Initialize pygame for potential rendering
    pygame.init()
    
    print(f"=> Starting TAAC Parallel Training")
    print(f"Environment: {env_name}")
    print(f"Job Name: {actual_job_name}")
    print(f"Training episodes: {config['training']['episodes']}")
    print(f"Parallel games: {num_parallel_games}")
    
    # Create sample environment to get info
    sample_env = TAACEnvironmentWrapper(
        env_name, 
        apply_wrappers=config['environment'].get('apply_wrappers', True),
        **config['environment'].get('env_kwargs', {})
    )
    
    # Environment info
    env_config = sample_env.env_info
    training_config = config['training'].copy()
    
    # Merge model configuration
    if 'model' in config:
        training_config.update(config['model'])
    
    print(f"Environment Configuration:")
    print(f"  - Agents: {env_config['num_agents']}")
    print(f"  - State size: {env_config['state_size']}")
    print(f"  - Action size: {env_config['action_size']}")
    print(f"  - Action type: discrete")
    
    # Initialize TAAC agent
    train_model = TAAC(env_config, training_config, mode="train")
    
    # Load model if specified
    if config.get('load_model'):
        if train_model.load_model(config['load_model']):
            print(f"=> Loaded model from: {config['load_model']}")
        else:
            print(f"=> Failed to load model from: {config['load_model']}")
    
    # Close sample environment
    sample_env.close()
    
    # Training parameters
    episodes = training_config['episodes']
    max_steps = training_config.get('max_steps_per_episode', 500)
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    
    # Initialize logger
    logger = TAACLogger(env_name, actual_job_name)
    
    # Training variables
    best_score = float('-inf')
    start_time = time.time()
    
    print(f"\n=> Starting parallel training for {episodes} episodes...")
    
    # Create the pool once, outside the training loop
    with mp_base.Pool(processes=num_parallel_games) as pool:
        for epoch in tqdm(range(episodes), desc="Training TAAC Parallel"):
            
            # Prepare arguments for parallel games
            if torch.cuda.is_available():
                args_list = [
                    (train_model,
                     i % torch.cuda.device_count(), env_name, config['environment'].get('env_kwargs', {}), max_steps)
                    for i in range(num_parallel_games)
                ]
            else:
                args_list = [
                    (train_model, env_name, config['environment'].get('env_kwargs', {}), max_steps)
                    for i in range(num_parallel_games)
                ]
            
            # Run parallel games
            results = pool.map(run_single_game_parallel, args_list)
            
            # Combine results
            combined_memories = []
            parallel_rewards = []
            parallel_entropies = []
            parallel_env_metrics = []
            
            for (metrics, memories) in results:
                episode_reward, normalized_entropy, env_metrics = metrics
                
                parallel_rewards.append(episode_reward)
                parallel_entropies.append(normalized_entropy)
                parallel_env_metrics.append(env_metrics)
                
                # Combine memories from all parallel runs
                if hasattr(memories, '__iter__'):
                    combined_memories.extend(memories)
                else:
                    combined_memories.append(memories)
            
            # Set combined memories back to the training model
            if hasattr(train_model, 'memories') and combined_memories:
                train_model.memories = combined_memories
            
            # Update the model with combined experiences
            similarity_loss = train_model.update()
            
            # Log parallel episode results
            logger.log_parallel_episodes(
                rewards=parallel_rewards,
                entropies=parallel_entropies,
                similarity_losses=[similarity_loss] * len(parallel_rewards),
                env_metrics_list=parallel_env_metrics
            )
            
            # Progress reporting
            if (epoch + 1) % log_interval == 0:
                logger.print_progress_report(epoch + 1, episodes, log_interval)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"{actual_job_name}_ep{epoch + 1}.pth")
                train_model.save_model(checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{actual_job_name}_final.pth")
    train_model.save_model(final_model_path)
    
    # Save final statistics
    logger.save_stats(log_dir, final_model_path=final_model_path)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n=> Parallel Training Complete!")
    print(f"Total Time: {format_time(total_time)}")
    print(f"Final Model: {final_model_path}")
    
    pygame.quit()
    return train_model 