#!/usr/bin/env python3
"""
Training script for TAAC algorithm on various PettingZoo environments
"""

import yaml
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

from TAAC import TAAC
from env_wrapper import TAACEnvironmentWrapper, create_env_config, ENV_CONFIGS


class ParallelEnvironmentManager:
    """Manages multiple environments running in parallel for faster training"""
    
    def __init__(self, env_name: str, env_kwargs: Dict[str, Any], num_parallel: int = 4):
        """
        Initialize parallel environment manager
        
        Args:
            env_name: Name of the environment
            env_kwargs: Environment configuration
            num_parallel: Number of parallel environments to run
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.num_parallel = num_parallel
        self.environments = []
        
        # Create parallel environments
        for i in range(num_parallel):
            env = TAACEnvironmentWrapper(env_name, **env_kwargs)
            self.environments.append(env)
            
        print(f"=> Created {num_parallel} parallel environments!")
        
    def reset_all(self) -> List[Tuple[List[np.ndarray], Dict]]:
        """Reset all parallel environments"""
        results = []
        for env in self.environments:
            states, info = env.reset()
            results.append((states, info))
        return results
    
    def step_all(self, actions_list: List[Dict[str, Any]]) -> List[Tuple[List[np.ndarray], List[float], bool, Dict]]:
        """Step all environments with their respective actions"""
        results = []
        for env, actions in zip(self.environments, actions_list):
            states, rewards, done, info = env.step(actions)
            results.append((states, rewards, done, info))
        return results
    
    def close_all(self):
        """Close all environments"""
        for env in self.environments:
            env.close()

class EpisodeLogger:
    """Enhanced logger for tracking episode metrics"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.max_heights = []  # For BoxJump specifically
        self.normalized_entropies = []  # Normalized entropy values (0-1 scale)
        self.episode_count = 0
        
    def log_episode(self, total_reward: float, episode_length: int, max_height: Optional[float] = None, normalized_entropy: Optional[float] = None):
        """Log a completed episode"""
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        if max_height is not None:
            self.max_heights.append(max_height)
            
        if normalized_entropy is not None:
            self.normalized_entropies.append(normalized_entropy)
            
        # Print episode summary
        height_str = f", Max Height: {max_height:.2f}" if max_height is not None else ""
        entropy_str = f", Entropy: {normalized_entropy:.3f}" if normalized_entropy is not None else ""
        print(f"> Episode {self.episode_count}: Reward: {total_reward:.2f}, Length: {episode_length}{height_str}{entropy_str}")
        
    def log_parallel_episodes(self, rewards: List[float], lengths: List[int], heights: List[Optional[float]] = None, entropies: List[Optional[float]] = None):
        """Log multiple episodes from parallel environments (taking averages)"""
        avg_reward = np.mean(rewards)
        avg_length = np.mean(lengths)
        avg_height = np.mean([h for h in heights if h is not None]) if heights and any(h is not None for h in heights) else None
        avg_entropy = np.mean([e for e in entropies if e is not None]) if entropies and any(e is not None for e in entropies) else None
        
        self.episode_count += 1
        self.episode_rewards.append(avg_reward)
        self.episode_lengths.append(avg_length)
        
        if avg_height is not None:
            self.max_heights.append(avg_height)
            
        if avg_entropy is not None:
            self.normalized_entropies.append(avg_entropy)
            
        # Print parallel episode summary
        height_str = f", Avg Max Height: {avg_height:.2f}" if avg_height is not None else ""
        entropy_str = f", Avg Entropy: {avg_entropy:.3f}" if avg_entropy is not None else ""
        print(f"> Parallel Episode {self.episode_count}: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}{height_str}{entropy_str}")
        print(f"   Individual Rewards: {[f'{r:.1f}' for r in rewards]}")
        if heights and any(h is not None for h in heights):
            print(f"   Individual Heights: {[f'{h:.1f}' if h is not None else 'N/A' for h in heights]}")
        if entropies and any(e is not None for e in entropies):
            print(f"   Individual Entropies: {[f'{e:.3f}' if e is not None else 'N/A' for e in entropies]}")
    
    def get_recent_stats(self, window: int = 10) -> Dict[str, float]:
        """Get statistics for recent episodes"""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
            
        if window == 0:
            return {}
            
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        stats = {
            'avg_reward': np.mean(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards)
        }
        
        if self.max_heights and len(self.max_heights) >= window:
            recent_heights = self.max_heights[-window:]
            stats.update({
                'avg_height': np.mean(recent_heights),
                'max_height': np.max(recent_heights)
            })
        
        if self.normalized_entropies and len(self.normalized_entropies) >= window:
            recent_entropies = [e for e in self.normalized_entropies[-window:] if e is not None]
            if recent_entropies:
                stats.update({
                    'avg_entropy': np.mean(recent_entropies),
                    'min_entropy': np.min(recent_entropies),
                    'max_entropy': np.max(recent_entropies)
                })
            
        return stats

def extract_height_from_states(states: List[np.ndarray], env_name: str) -> Optional[float]:
    """Extract tower height from environment states (BoxJump specific)"""
    if env_name != 'boxjump':
        return None
        
    # For BoxJump, the highest y-coordinate is at index 11 in each agent's observation
    # We take the maximum across all agents
    try:
        heights = []
        for state in states:
            if len(state) > 11:  # Ensure observation has height data
                heights.append(state[11])  # Index 11 contains highest y-coordinate
        return max(heights) if heights else None
    except (IndexError, ValueError):
        return None


def normalize_entropy(entropy_dict: Dict[str, float], action_space_type: str, num_actions: int = None, action_size: int = None) -> float:
    """
    Normalize entropy to 0-1 scale where:
    - 1.0 = Maximum entropy (completely random behavior)
    - 0.0 = Minimum entropy (completely deterministic behavior)
    
    Args:
        entropy_dict: Dictionary of entropy values per agent
        action_space_type: "discrete" or "continuous"
        num_actions: Number of discrete actions (for discrete spaces)
        action_size: Dimensionality of continuous actions (for continuous spaces)
        
    Returns:
        Normalized entropy value between 0 and 1
    """
    if not entropy_dict:
        return None
    
    # Average entropy across all agents
    avg_entropy = np.mean(list(entropy_dict.values()))
    
    if action_space_type == "discrete":
        # For discrete actions, max entropy = log(num_actions)
        if num_actions is None or num_actions <= 1:
            return None
        max_entropy = np.log(num_actions)
        min_entropy = 0.0
    else:  # continuous
        # For continuous actions, we use empirical bounds
        # Typical entropy range for normal distributions is roughly [-10, 10] per dimension
        if action_size is None:
            action_size = 1
        max_entropy = 2.5 * action_size  # Approximate upper bound
        min_entropy = -2.5 * action_size  # Approximate lower bound
    
    # Normalize to [0, 1] range
    if max_entropy == min_entropy:
        return 0.5  # Fallback when range is zero
    
    normalized = (avg_entropy - min_entropy) / (max_entropy - min_entropy)
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, normalized))

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_default_config(env_name: str) -> Dict[str, Any]:
    """Create default configuration for an environment"""
    base_config = {
        'environment': {
            'name': env_name,
            'env_kwargs': {},
            'apply_wrappers': True
        },
        'training': {
            'episodes': 1000,
            'max_steps_per_episode': 500,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'epsilon_clip': 0.2,
            'K_epochs': 10,
            'c_entropy': 0.01,
            'max_grad_norm': 0.5,
            'c_value': 0.5,
            'lam': 0.95,
            'batch_size': 64,
            'min_learning_rate': 1e-6,
            'similarity_loss_coef': 0.1
        },
        'logging': {
            'log_interval': 10,
            'save_interval': 100,
            'eval_interval': 50,
            'save_best_model': True
        },
        'model': {
            'num_heads': 4,
            'embedding_dim': 256,
            'hidden_size': 526,
            'save_dir': 'files/Models',
            'model_name': f'TAAC_{env_name}'
        }
    }
    
    # Override with environment-specific configs if available
    if env_name in ENV_CONFIGS:
        env_config = ENV_CONFIGS[env_name]
        if 'env_kwargs' in env_config:
            base_config['environment']['env_kwargs'].update(env_config['env_kwargs'])
        if 'training_config' in env_config:
            base_config['training'].update(env_config['training_config'])
    
    return base_config


def evaluate_agent(taac_agent: TAAC, env: TAACEnvironmentWrapper, num_episodes: int = 5) -> Dict[str, float]:
    """Evaluate the trained agent"""
    taac_agent.mode = "test"
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        states, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            actions, _ = taac_agent.get_actions(states)
            states, rewards, done, _ = env.step(actions)
            
            episode_reward += sum(rewards)
            step_count += 1
            
            if step_count >= 1000:  # Prevent infinite episodes
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
    
    taac_agent.mode = "train"
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def train_taac(config: Dict[str, Any], save_dir: str = None) -> TAAC:
    """
    Train TAAC agent on specified environment (single environment)
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save results
    
    Returns:
        Trained TAAC agent
    """
    # Setup
    env_name = config['environment']['name']
    env_kwargs = config['environment'].get('env_kwargs', {})
    apply_wrappers = config['environment'].get('apply_wrappers', True)
    
    print(f"Setting up single environment: {env_name}")
    env = TAACEnvironmentWrapper(env_name, apply_wrappers=apply_wrappers, **env_kwargs)
    
    # Create environment configuration for TAAC
    env_config = env.env_info
    training_config = config['training'].copy()
    
    # Merge model configuration into training config
    if 'model' in config:
        training_config.update(config['model'])
        print(f">>  Model Configuration:")
        print(f"  - Embedding Dim: {training_config.get('embedding_dim', 256)}")
        print(f"  - Hidden Size: {training_config.get('hidden_size', 526)}")
        print(f"  - Attention Heads: {training_config.get('num_heads', 4)}")
    
    print(f"Environment Info:")
    print(f"  - Agents: {env_config['num_agents']}")
    print(f"  - State size: {env_config['state_size']}")
    print(f"  - Action size: {env_config['action_size']}")
    print(f"  - Action type: {env_config['action_space_type']}")
    
    # Initialize TAAC agent
    taac_agent = TAAC(env_config, training_config, mode="train")
    
    # Training variables
    episodes = training_config['episodes']
    max_steps = training_config.get('max_steps_per_episode', 500)
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    eval_interval = config['logging']['eval_interval']
    
    # Enhanced logging
    logger = EpisodeLogger(env_name)
    similarity_losses = []
    eval_scores = []
    best_score = float('-inf')
    
    print(f"\nStarting single-environment training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset environment and agent memory
        states, _ = env.reset()
        taac_agent.memory_prep(env.num_agents)
        
        episode_reward = 0
        step_count = 0
        done = False
        max_height_this_episode = None
        
        # Episode loop
        while not done and step_count < max_steps:
            # Get actions from all agents
            actions, entropies = taac_agent.get_actions(states)
            
            # Step environment
            next_states, rewards, done, info = env.step(actions)
            
            # Extract height information (BoxJump specific)
            current_height = extract_height_from_states(next_states, env_name)
            if current_height is not None:
                if max_height_this_episode is None or current_height > max_height_this_episode:
                    max_height_this_episode = current_height
            
            # Store rewards
            taac_agent.store_rewards(rewards, done)
            
            episode_reward += sum(rewards)
            step_count += 1
            states = next_states
        
        # Update agent after episode
        if episode > 0:  # Skip first episode for memory initialization
            similarity_loss = taac_agent.update()
            if similarity_loss is not None:
                similarity_losses.append(similarity_loss)
        
        # Log episode
        logger.log_episode(episode_reward, step_count, max_height_this_episode)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            stats = logger.get_recent_stats(log_interval)
            elapsed_time = time.time() - start_time
            
            print(f"\n> Episode {episode + 1}/{episodes} Progress Report:")
            print(f"  *  Time Elapsed: {elapsed_time:.1f}s")
            print(f"  > Avg Reward (last {min(log_interval, len(logger.episode_rewards))}): {stats.get('avg_reward', 0):.2f}")
            print(f"  - Avg Length: {stats.get('avg_length', 0):.1f}")
            if 'avg_height' in stats:
                print(f"  #  Avg Tower Height: {stats['avg_height']:.2f}")
                print(f"  * Max Tower Height: {stats['max_height']:.2f}")
            if similarity_losses:
                print(f"  ~ Similarity Loss: {similarity_losses[-1]:.4f}")
            print(f"  => Training Speed: {(episode + 1)/elapsed_time:.2f} episodes/sec")
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            print("\n* Running evaluation...")
            eval_results = evaluate_agent(taac_agent, env, num_episodes=5)
            eval_scores.append(eval_results['mean_reward'])
            
            print(f"  > Eval Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  - Eval Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            # Save best model
            if config['logging']['save_best_model'] and eval_results['mean_reward'] > best_score:
                best_score = eval_results['mean_reward']
                model_name = f"{config['model']['model_name']}_best"
                taac_agent.save_model(model_name)
                print(f"  * New best model saved: {model_name}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            model_name = f"{config['model']['model_name']}_ep{episode + 1}"
            taac_agent.save_model(model_name)
    
    # Save final model
    final_model_name = f"{config['model']['model_name']}_final"
    taac_agent.save_model(final_model_name)
    
    # Save training statistics
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save statistics
        stats = {
            'episode_rewards': logger.episode_rewards,
            'episode_lengths': logger.episode_lengths,
            'max_heights': logger.max_heights,
            'similarity_losses': similarity_losses,
            'eval_scores': eval_scores,
            'config': config
        }
        
        # Convert numpy types to native Python types for JSON serialization  
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Convert all stats to JSON-serializable format
        json_stats = convert_numpy_types(stats)
        
        stats_path = os.path.join(save_dir, f'training_stats_{env_name}_single.json')
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        # Plot training curves
        plot_enhanced_training_curves(logger, eval_scores, save_dir, env_name)
    
    env.close()
    print(f"\n! Single-environment training completed! Final model saved as: {final_model_name}")
    
    return taac_agent


def train_taac_parallel(config: Dict[str, Any], save_dir: str = None, num_parallel: int = 4) -> TAAC:
    """
    Train TAAC with parallel environments for faster training
    
    Args:
        config: Training configuration
        save_dir: Directory to save results
        num_parallel: Number of parallel environments
    
    Returns:
        Trained TAAC agent
    """
    env_name = config['environment']['name']
    env_kwargs = config['environment']['env_kwargs']
    training_config = config['training'].copy()
    
    # Merge model configuration into training config
    if 'model' in config:
        training_config.update(config['model'])
        print(f">>  Model Configuration:")
        print(f"  - Embedding Dim: {training_config.get('embedding_dim', 256)}")
        print(f"  - Hidden Size: {training_config.get('hidden_size', 526)}")
        print(f"  - Attention Heads: {training_config.get('num_heads', 4)}")
    
    print(f"=> Starting PARALLEL training with {num_parallel} environments")
    print(f"Environment: {env_name}")
    print(f"Training episodes: {training_config['episodes']}")
    
    # Create parallel environment manager
    parallel_manager = ParallelEnvironmentManager(env_name, env_kwargs, num_parallel)
    
    # Get environment configuration from first environment
    sample_env = parallel_manager.environments[0]
    env_config = sample_env.env_info
    
    print(f"Environment Configuration:")
    print(f"  - Agents: {env_config['num_agents']}")
    print(f"  - State size: {env_config['state_size']}")
    print(f"  - Action size: {env_config['action_size']}")
    print(f"  - Action type: {env_config['action_space_type']}")
    
    # Initialize TAAC agent
    taac_agent = TAAC(env_config, training_config, mode="train")
    
    # Training variables
    episodes = training_config['episodes']
    max_steps = training_config.get('max_steps_per_episode', 500)
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    eval_interval = config['logging']['eval_interval']
    
    # Enhanced logging
    logger = EpisodeLogger(env_name)
    similarity_losses = []
    eval_scores = []
    eval_episodes = []  # Track which episodes evaluations happened at
    best_score = float('-inf')
    
    print(f"\n> Starting parallel training for {episodes} episodes...")
    start_time = time.time()
    
    # Parallel training loop
    episode = 0
    while episode < episodes:
        # Reset all environments
        reset_results = parallel_manager.reset_all()
        
        # Prepare agent memory for all environments
        for _ in range(num_parallel):
            taac_agent.memory_prep(env_config['num_agents'])
        
        # Run episodes in parallel
        parallel_rewards = []
        parallel_lengths = []
        parallel_heights = []
        parallel_entropies = []
        
        for env_idx in range(num_parallel):
            if episode + env_idx >= episodes:
                break
                
            states, _ = reset_results[env_idx]
            episode_reward = 0
            step_count = 0
            done = False
            max_height_this_episode = None
            episode_entropies = []  # Collect entropies for this episode
            
            # Episode loop for this environment
            while not done and step_count < max_steps:
                # Get actions from agent
                actions, entropies = taac_agent.get_actions(states)
                
                # Collect entropy for normalization (sample every 10 steps to reduce noise)
                if step_count % 10 == 0:
                    episode_entropies.append(entropies)
                
                # Step environment
                next_states, rewards, done, info = parallel_manager.environments[env_idx].step(actions)
                
                # Extract height information (BoxJump specific)
                current_height = extract_height_from_states(next_states, env_name)
                if current_height is not None:
                    if max_height_this_episode is None or current_height > max_height_this_episode:
                        max_height_this_episode = current_height
                
                # Store rewards for training
                taac_agent.store_rewards(rewards, done)
                
                episode_reward += sum(rewards)
                step_count += 1
                states = next_states
                
                # Reset environment when done
                if done:
                    states, _ = parallel_manager.environments[env_idx].reset()
            
            # Calculate normalized entropy for this episode (average across sampled steps)
            if episode_entropies:
                # Average entropy values across agents and time steps
                avg_entropy_dict = {}
                for agent_key in episode_entropies[0].keys():
                    avg_entropy_dict[agent_key] = np.mean([ent[agent_key] for ent in episode_entropies])
                
                normalized_entropy = normalize_entropy(
                    avg_entropy_dict, 
                    env_config['action_space_type'], 
                    env_config.get('num_actions', env_config.get('action_size'))
                )
            else:
                normalized_entropy = None
            
            parallel_rewards.append(episode_reward)
            parallel_lengths.append(step_count)
            parallel_heights.append(max_height_this_episode)
            parallel_entropies.append(normalized_entropy)
        
        # Update agent after batch of parallel episodes
        if episode > 0:  # Skip first episode for memory initialization
            similarity_loss = taac_agent.update()
            if similarity_loss is not None:
                similarity_losses.append(similarity_loss)
        
        # Log parallel episodes
        actual_parallel_count = min(num_parallel, episodes - episode)
        logger.log_parallel_episodes(
            parallel_rewards[:actual_parallel_count], 
            parallel_lengths[:actual_parallel_count],
            parallel_heights[:actual_parallel_count],
            parallel_entropies[:actual_parallel_count]
        )
        
        episode += actual_parallel_count
        
        # Enhanced logging with recent statistics
        if episode % log_interval == 0:
            stats = logger.get_recent_stats(log_interval)
            elapsed_time = time.time() - start_time
            
            print(f"\n> Episode {episode}/{episodes} Progress Report:")
            print(f"  *  Time Elapsed: {elapsed_time:.1f}s")
            print(f"  > Avg Reward (last {min(log_interval, len(logger.episode_rewards))}): {stats.get('avg_reward', 0):.2f}")
            print(f"  - Avg Length: {stats.get('avg_length', 0):.1f}")
            if 'avg_height' in stats:
                print(f"  #  Avg Tower Height: {stats['avg_height']:.2f}")
                print(f"  * Max Tower Height: {stats['max_height']:.2f}")
            if 'avg_entropy' in stats:
                print(f"  & Avg Entropy: {stats['avg_entropy']:.3f}")
            if similarity_losses:
                print(f"  ~ Similarity Loss: {similarity_losses[-1]:.4f}")
            print(f"  => Training Speed: {episode/elapsed_time:.2f} episodes/sec")
            
        # Evaluation
        if episode % eval_interval == 0:
            print("\n* Running evaluation...")
            eval_results = evaluate_agent(taac_agent, parallel_manager.environments[0], num_episodes=5)
            eval_scores.append(eval_results['mean_reward'])
            eval_episodes.append(episode)  # Track which episode this evaluation happened at
            
            print(f"  > Eval Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  - Eval Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            # Save best model
            if config['logging']['save_best_model'] and eval_results['mean_reward'] > best_score:
                best_score = eval_results['mean_reward']
                model_name = f"{config['model']['model_name']}_best"
                taac_agent.save_model(model_name)
                print(f"  * New best model saved: {model_name}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            model_name = f"{config['model']['model_name']}_ep{episode}"
            taac_agent.save_model(model_name)
    
    # Clean up parallel environments
    parallel_manager.close_all()
    
    # Save final model
    final_model_name = f"{config['model']['model_name']}_final"
    taac_agent.save_model(final_model_name)
    
    # Save enhanced training statistics
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Enhanced statistics including height data
        stats = {
            'episode_rewards': logger.episode_rewards,
            'episode_lengths': logger.episode_lengths,
            'max_heights': logger.max_heights,
            'normalized_entropies': logger.normalized_entropies,
            'similarity_losses': similarity_losses,
            'eval_scores': eval_scores,
            'eval_episodes': eval_episodes,
            'config': config,
            'num_parallel_envs': num_parallel,
            'total_episodes': episode
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Convert all stats to JSON-serializable format
        json_stats = convert_numpy_types(stats)
        
        stats_path = os.path.join(save_dir, f'training_stats_{env_name}_parallel.json')
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        # Plot enhanced training curves
        plot_enhanced_training_curves(logger, eval_scores, save_dir, env_name, eval_episodes)
    
    print(f"\n! Parallel training completed! Final model saved as: {final_model_name}")
    print(f"> Total episodes: {episode}")
    print(f"* Average training speed: {episode/(time.time() - start_time):.2f} episodes/sec")
    
    return taac_agent

def plot_enhanced_training_curves(logger: EpisodeLogger, eval_scores: List[float], save_dir: str, env_name: str, eval_episodes: List[int] = None):
    """Plot enhanced training curves with height information"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'TAAC Parallel Training Results - {env_name}', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(logger.episode_rewards, alpha=0.7, color='blue', label='Episode Rewards')
    if len(logger.episode_rewards) > 50:
        window = min(50, len(logger.episode_rewards) // 10)
        moving_avg = np.convolve(logger.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(logger.episode_rewards)), moving_avg, color='red', linewidth=2, label='Moving Average')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(logger.episode_lengths, alpha=0.7, color='green', label='Episode Lengths')
    if len(logger.episode_lengths) > 50:
        window = min(50, len(logger.episode_lengths) // 10)
        moving_avg = np.convolve(logger.episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(logger.episode_lengths)), moving_avg, color='orange', linewidth=2, label='Moving Average')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Tower heights (BoxJump specific)
    if logger.max_heights:
        axes[0, 2].plot(logger.max_heights, alpha=0.7, color='purple', label='Max Tower Height')
        if len(logger.max_heights) > 20:
            window = min(20, len(logger.max_heights) // 5)
            moving_avg = np.convolve(logger.max_heights, np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(logger.max_heights)), moving_avg, color='darkviolet', linewidth=2, label='Moving Average')
        axes[0, 2].set_title('Maximum Tower Height per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Height')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Height Data\n(Not BoxJump)', 
                       transform=axes[0, 2].transAxes, ha='center', va='center', fontsize=12)
        axes[0, 2].set_title('Tower Height (N/A)')
    
    # Evaluation scores
    if eval_scores:
        # Use provided eval_episodes if available, otherwise calculate them
        if eval_episodes is not None and len(eval_episodes) == len(eval_scores):
            plot_eval_episodes = eval_episodes
        else:
            # Fallback: calculate evaluation episode indices
            if len(logger.episode_rewards) >= 50:
                plot_eval_episodes = np.arange(50, len(logger.episode_rewards) + 1, 50)[:len(eval_scores)]
                # Ensure we have matching dimensions
                if len(plot_eval_episodes) == 0 and len(eval_scores) > 0:
                    # Fallback: create episode indices based on eval_scores length
                    plot_eval_episodes = np.linspace(50, len(logger.episode_rewards), len(eval_scores))
            else:
                # For short training runs, just create sequential indices
                plot_eval_episodes = np.arange(1, len(eval_scores) + 1)
        
        # Ensure we have matching dimensions before plotting
        if len(plot_eval_episodes) == len(eval_scores) and len(eval_scores) > 0:
            axes[1, 0].plot(plot_eval_episodes, eval_scores, marker='o', color='red', linewidth=2, markersize=6)
            axes[1, 0].set_title('Evaluation Scores')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Mean Eval Reward')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, f'Eval Data Mismatch\n{len(plot_eval_episodes)} episodes vs {len(eval_scores)} scores', 
                           transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=10)
            axes[1, 0].set_title('Evaluation Scores (Error)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Evaluation Data', 
                       transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Evaluation Scores (N/A)')
    
    # Reward distribution
    axes[1, 1].hist(logger.episode_rewards, bins=30, alpha=0.7, color='cyan', edgecolor='black')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    # Normalized Entropy (Policy Learning Progress)
    if logger.normalized_entropies:
        valid_entropies = [e for e in logger.normalized_entropies if e is not None]
        if valid_entropies:
            axes[1, 2].plot(valid_entropies, alpha=0.7, color='orange', label='Normalized Entropy', linewidth=2)
            if len(valid_entropies) > 20:
                window = min(20, len(valid_entropies) // 5)
                moving_avg = np.convolve(valid_entropies, np.ones(window)/window, mode='valid')
                axes[1, 2].plot(range(window-1, len(valid_entropies)), moving_avg, color='red', linewidth=2, label='Moving Average')
            axes[1, 2].set_title('Policy Entropy (Learning Progress)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Normalized Entropy (1=random, 0=deterministic)')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Valid Entropy Data', 
                           transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=12)
            axes[1, 2].set_title('Entropy (No Data)')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Entropy Data', 
                       transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Entropy (N/A)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enhanced_training_curves_{env_name}_parallel.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main_with_args(args=None):
    """Main function that can accept args directly or parse from command line"""
    if args is None:
        # Parse from command line if no args provided
        parser = argparse.ArgumentParser(description='Train TAAC on PettingZoo environments')
        parser.add_argument('--env', type=str, required=True, 
                           help='Environment name (e.g., cooking_zoo, boxjump, mpe_simple_spread)')
        parser.add_argument('--config', type=str, default=None,
                           help='Path to config file (if not provided, uses default)')
        parser.add_argument('--save_dir', type=str, default='experiments',
                           help='Directory to save results')
        parser.add_argument('--episodes', type=int, default=None,
                           help='Number of episodes to train (overrides config)')
        parser.add_argument('--eval_only', action='store_true',
                           help='Only run evaluation with existing model')
        parser.add_argument('--model_path', type=str, default=None,
                           help='Path to model for evaluation')
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--gamma', type=float, default=None)
        parser.add_argument('--log_interval', type=int, default=None)
        parser.add_argument('--save_interval', type=int, default=None)
        parser.add_argument('--eval_interval', type=int, default=None)
        parser.add_argument('--model_save_path', type=str, default=None)
        parser.add_argument('--render', action='store_true')
        parser.add_argument('--num_parallel', type=int, default=4,
                           help='Number of parallel environments for training')
        
        args = parser.parse_args()
    
    # Use config from args if available (passed from run_taac.py)
    if hasattr(args, 'config') and isinstance(args.config, dict):
        config = args.config
        experiment_dir = args.model_save_path or 'experiments'
    else:
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(getattr(args, 'save_dir', 'experiments'), f"{args.env}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Load or create configuration
        if hasattr(args, 'config') and args.config:
            config = load_config(args.config)
        else:
            config = create_default_config(args.env)
            config_path = os.path.join(experiment_dir, 'config.yaml')
            save_config(config, config_path)
            print(f"Using default config saved to: {config_path}")
    
    # Override config values with command line arguments if provided
    if hasattr(args, 'episodes') and args.episodes:
        config['training']['episodes'] = args.episodes
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if hasattr(args, 'gamma') and args.gamma:
        config['training']['gamma'] = args.gamma
    
    # Update environment name in config
    if hasattr(args, 'env'):
        config['environment']['name'] = args.env
    
    if hasattr(args, 'eval_only') and args.eval_only:
        # Evaluation only
        if not hasattr(args, 'model_path') or not args.model_path:
            raise ValueError("Model path required for evaluation")
        
        print("Running evaluation only...")
        env_name = config['environment']['name']
        env = TAACEnvironmentWrapper(env_name, **config['environment'].get('env_kwargs', {}))
        
        # Create TAAC agent and load model
        env_config = env.env_info
        taac_agent = TAAC(env_config, config['training'], mode="test")
        taac_agent.load_model(args.model_path, test=True)
        
        # Run evaluation
        eval_results = evaluate_agent(taac_agent, env, num_episodes=20)
        print(f"\nEvaluation Results:")
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
        
        env.close()
    else:
        # Training
        # Ensure experiment directory exists
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Choose training mode based on num_parallel parameter
        if args.num_parallel > 1:
            print(f"=> Using PARALLEL training with {args.num_parallel} environments")
            taac_agent = train_taac_parallel(config, experiment_dir, args.num_parallel)
        else:
            print(f">> Using SINGLE environment training")
            taac_agent = train_taac(config, experiment_dir)
            
        print(f"\nExperiment results saved to: {experiment_dir}")


def main():
    """Wrapper for command line usage"""
    main_with_args()


if __name__ == "__main__":
    main() 