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
from typing import Dict, Any, List
import time
from datetime import datetime
import json

from TAAC import TAAC
from env_wrapper import TAACEnvironmentWrapper, create_env_config, ENV_CONFIGS


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
    Train TAAC agent on specified environment
    
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
    
    print(f"Setting up environment: {env_name}")
    env = TAACEnvironmentWrapper(env_name, apply_wrappers=apply_wrappers, **env_kwargs)
    
    # Create environment configuration for TAAC
    env_config = env.env_info
    training_config = config['training']
    
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
    
    # Tracking variables
    episode_rewards = []
    episode_lengths = []
    similarity_losses = []
    eval_scores = []
    best_score = float('-inf')
    
    print(f"\nStarting training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset environment and agent memory
        states, _ = env.reset()
        taac_agent.memory_prep(env.num_agents)
        
        episode_reward = 0
        step_count = 0
        done = False
        
        # Episode loop
        while not done and step_count < max_steps:
            # Get actions from all agents
            actions, entropies = taac_agent.get_actions(states)
            
            # Step environment
            next_states, rewards, done, info = env.step(actions)
            
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
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last {log_interval}): {avg_reward:.2f}")
            print(f"  Avg Length (last {log_interval}): {avg_length:.1f}")
            print(f"  Time Elapsed: {elapsed_time:.1f}s")
            if similarity_losses:
                print(f"  Similarity Loss: {similarity_losses[-1]:.4f}")
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            print("Running evaluation...")
            eval_results = evaluate_agent(taac_agent, env, num_episodes=5)
            eval_scores.append(eval_results['mean_reward'])
            
            print(f"  Eval Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Eval Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            # Save best model
            if config['logging']['save_best_model'] and eval_results['mean_reward'] > best_score:
                best_score = eval_results['mean_reward']
                model_name = f"{config['model']['model_name']}_best"
                taac_agent.save_model(model_name)
                print(f"  New best model saved: {model_name}")
        
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
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'similarity_losses': similarity_losses,
            'eval_scores': eval_scores,
            'config': config
        }
        
        stats_path = os.path.join(save_dir, f'training_stats_{env_name}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Plot training curves
        plot_training_curves(episode_rewards, episode_lengths, eval_scores, save_dir, env_name)
    
    env.close()
    print(f"\nTraining completed! Final model saved as: {final_model_name}")
    
    return taac_agent


def plot_training_curves(episode_rewards: List[float], episode_lengths: List[int], 
                        eval_scores: List[float], save_dir: str, env_name: str):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'TAAC Training Results - {env_name}', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.7, color='blue')
    if len(episode_rewards) > 50:
        # Plot moving average
        window = min(50, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.7, color='green')
    if len(episode_lengths) > 50:
        window = min(50, len(episode_lengths) // 10)
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg, color='orange', linewidth=2)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Evaluation scores
    if eval_scores:
        eval_episodes = np.arange(50, len(episode_rewards) + 1, 50)[:len(eval_scores)]
        axes[1, 0].plot(eval_episodes, eval_scores, marker='o', color='purple')
        axes[1, 0].set_title('Evaluation Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Mean Eval Reward')
        axes[1, 0].grid(True)
    
    # Reward distribution
    axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, color='cyan')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{env_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
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
    
    args = parser.parse_args()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"{args.env}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config(args.env)
        config_path = os.path.join(experiment_dir, 'config.yaml')
        save_config(config, config_path)
        print(f"Using default config saved to: {config_path}")
    
    # Override episodes if specified
    if args.episodes:
        config['training']['episodes'] = args.episodes
    
    # Update environment name in config
    config['environment']['name'] = args.env
    
    if args.eval_only:
        # Evaluation only
        if not args.model_path:
            raise ValueError("Model path required for evaluation")
        
        print("Running evaluation only...")
        env = TAACEnvironmentWrapper(args.env, **config['environment'].get('env_kwargs', {}))
        
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
        taac_agent = train_taac(config, experiment_dir)
        print(f"\nExperiment results saved to: {experiment_dir}")


if __name__ == "__main__":
    main() 