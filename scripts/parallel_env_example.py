#!/usr/bin/env python3
"""
Example script demonstrating true parallel environment execution using multiprocessing.
This is different from the existing parallel implementation in TAAC which runs
multiple environments sequentially in the same process.

Usage:
    python scripts/parallel_env_example.py --config configs/boxjump.yaml --num_processes 4
"""

import argparse
import yaml
import os
import sys
import time
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Add AI directory to path
sys.path.append(str(Path(__file__).parent.parent / "AI"))

from env_wrapper import TAACEnvironmentWrapper, create_env_config
from TAAC import TAAC


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_episode(args):
    """
    Run a single episode in a separate process.
    
    Args:
        args: Tuple containing (process_id, config, model_path, gpu_id, render)
        
    Returns:
        Tuple containing episode statistics and collected experiences
    """
    process_id, config, model_path, gpu_id, render = args
    
    # Set different random seed for each process
    torch.manual_seed(process_id)
    np.random.seed(process_id)
    
    # Set device for this process
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Create environment
    env_name = config['environment']['name']
    env_kwargs = config['environment']['env_kwargs']
    
    # Add render mode if specified
    if render:
        env_kwargs['render_mode'] = 'human'
    
    env = TAACEnvironmentWrapper(env_name, **env_kwargs)
    env_config = env.env_info
    
    # Create agent
    training_config = config['training'].copy()
    if 'model' in config:
        training_config.update(config['model'])
    
    agent = TAAC(env_config, training_config, mode="train")
    
    # Load model if path provided
    if model_path:
        agent.load_model(model_path)
    
    # Run episode
    states, _ = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    max_steps = config['training'].get('max_steps_per_episode', 500)
    
    # Initialize memory for this episode
    agent.memory_prep(env_config['num_agents'])
    
    # Track entropy for this episode
    entropies = []
    
    # Extract environment-specific metrics
    if env_name == 'boxjump':
        max_height = 0
    else:
        max_height = None
    
    while not done and step_count < max_steps:
        # Get actions from agent
        actions, action_entropies = agent.get_actions(states)
        
        # Collect entropy (every 10 steps to reduce noise)
        if step_count % 10 == 0:
            entropies.append(action_entropies)
        
        # Step environment
        next_states, rewards, done, info = env.step(actions)
        
        # Track environment-specific metrics
        if env_name == 'boxjump' and 'max_height' in info:
            max_height = max(max_height, info['max_height'])
        
        # Store rewards for training
        agent.store_rewards(rewards, done)
        
        episode_reward += sum(rewards)
        step_count += 1
        states = next_states
    
    # Close environment
    env.close()
    
    # Calculate normalized entropy
    if entropies:
        # Average entropy values across agents and time steps
        avg_entropy_dict = {}
        for agent_key in entropies[0].keys():
            avg_entropy_dict[agent_key] = np.mean([ent[agent_key] for ent in entropies])
        
        # Normalize entropy (0-1 scale)
        if env_config['action_space_type'] == "discrete":
            num_actions = env_config.get('num_actions', env_config.get('action_size'))
            max_entropy = np.log(num_actions) if num_actions > 1 else 1
        else:  # Continuous
            # For continuous actions, use a different approach
            action_size = env_config.get('action_size', 1)
            max_entropy = 0.5 * np.log(2 * np.pi * np.e) * action_size
        
        # Average entropy across all agents
        avg_entropy = np.mean(list(avg_entropy_dict.values()))
        normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = None
    
    # Return episode statistics and agent memories
    return {
        'reward': episode_reward,
        'length': step_count,
        'max_height': max_height,
        'normalized_entropy': normalized_entropy,
        'process_id': process_id
    }, agent.memories


def run_parallel_training(config, num_processes=4, num_episodes=None, model_path=None, render=False):
    """
    Run training using multiple processes for true parallel execution.
    
    Args:
        config: Configuration dictionary
        num_processes: Number of parallel processes to use
        num_episodes: Total number of episodes to run (overrides config)
        model_path: Path to model to load (optional)
        render: Whether to render the environment
    """
    # Get configuration
    env_name = config['environment']['name']
    episodes = num_episodes or config['training']['episodes']
    
    # Create master agent for updates
    sample_env = TAACEnvironmentWrapper(env_name, **config['environment']['env_kwargs'])
    env_config = sample_env.env_info
    sample_env.close()
    
    training_config = config['training'].copy()
    if 'model' in config:
        training_config.update(config['model'])
    
    master_agent = TAAC(env_config, training_config, mode="train")
    if model_path:
        master_agent.load_model(model_path)
    
    # Determine GPU usage
    use_gpu = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if use_gpu else 0
    
    # If rendering, we can only use one process
    if render:
        print("Rendering enabled - using single process mode")
        num_processes = 1
    
    print(f"Starting parallel training with {num_processes} processes")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"GPUs available: {num_gpus}")
    
    # Training loop
    episode = 0
    start_time = time.time()
    
    # Use Pool for process management
    with mp.Pool(processes=num_processes) as pool:
        while episode < episodes:
            # Determine how many episodes to run in this batch
            batch_size = min(num_processes, episodes - episode)
            
            # Prepare arguments for each process
            args_list = []
            for i in range(batch_size):
                process_id = episode + i
                # Assign GPUs in round-robin fashion if available
                gpu_id = (process_id % num_gpus) if use_gpu else None
                args_list.append((process_id, config, model_path, gpu_id, render and i == 0))
            
            # Run episodes in parallel
            results = pool.map(run_single_episode, args_list)
            
            # Process results
            batch_rewards = []
            batch_lengths = []
            batch_heights = []
            batch_entropies = []
            combined_memories = []
            
            for stats, memories in results:
                batch_rewards.append(stats['reward'])
                batch_lengths.append(stats['length'])
                if stats['max_height'] is not None:
                    batch_heights.append(stats['max_height'])
                if stats['normalized_entropy'] is not None:
                    batch_entropies.append(stats['normalized_entropy'])
                combined_memories.extend(memories)
            
            # Update master agent with combined experiences
            master_agent.memories = combined_memories
            similarity_loss = master_agent.update()
            
            # Log results
            avg_reward = np.mean(batch_rewards)
            avg_length = np.mean(batch_lengths)
            avg_height = np.mean(batch_heights) if batch_heights else None
            avg_entropy = np.mean(batch_entropies) if batch_entropies else None
            
            # Update progress
            episode += batch_size
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            
            # Print progress
            height_str = f", Height: {avg_height:.2f}" if avg_height is not None else ""
            entropy_str = f", Entropy: {avg_entropy:.3f}" if avg_entropy is not None else ""
            sim_loss_str = f", Sim Loss: {similarity_loss:.4f}" if similarity_loss is not None else ""
            
            print(f"Episode {episode}/{episodes}: Reward: {avg_reward:.2f}, Length: {avg_length:.1f}{height_str}{entropy_str}{sim_loss_str}, Speed: {eps_per_sec:.2f} eps/sec")
            
            # Save model periodically
            if episode % config['logging']['save_interval'] == 0:
                model_name = f"{config['model']['model_name']}_ep{episode}"
                master_agent.save_model(model_name)
                print(f"Saved model: {model_name}")
    
    # Save final model
    final_model_name = f"{config['model']['model_name']}_final"
    master_agent.save_model(final_model_name)
    print(f"Saved final model: {final_model_name}")
    
    total_time = time.time() - start_time
    print(f"Training complete! Total time: {total_time:.1f}s, Avg speed: {episodes/total_time:.2f} eps/sec")
    
    return master_agent


def main():
    parser = argparse.ArgumentParser(description="True parallel environment execution example")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to run')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model to load')
    parser.add_argument('--render', action='store_true', help='Render environment (forces single process)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run parallel training
    run_parallel_training(
        config=config,
        num_processes=args.num_processes,
        num_episodes=args.episodes,
        model_path=args.model_path,
        render=args.render
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA support
    main() 