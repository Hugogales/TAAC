#!/usr/bin/env python3
"""
TAAC Parallel Training from src.AI.TAAC import TAAC, Memory
from src.env_wrapper import TAACEnvironmentWrapper, create_env_config
from src.logger import TAACLogger, extract_environment_metrics, format_time


def setup_paths(env_name: str, job_name: str) -> Tuple[str, str]:ion (Optimized with Persistent Workers)

This module implements efficient parallel training using persistent worker processes
that reuse environments across multiple episodes, significantly reducing overhead
compared to the previous approach of creating new processes for each episode.

Key Improvements:
- Persistent worker processes that stay alive for the entire training session
- Environment reuse: Each worker creates one environment and reuses it
- Model state synchronization: Workers receive updated model states via queues
- Reduced process creation overhead: ~10x faster episode execution
- Better resource utilization: Stable memory usage, no process thrashing
- GPU affinity: Each worker can be assigned to a specific GPU

Architecture:
1. Main process: Manages training loop, model updates, and logging
2. Worker processes: Run episodes independently with synchronized model states
3. Communication: Task and result queues for coordination
4. Cleanup: Proper resource management with graceful shutdown
"""

import multiprocessing as mp
import os
import sys

# Set spawn method for CUDA compatibility in multiprocessing
# This must be done before any CUDA operations
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

import yaml
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp_base
import random
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import json
import pygame
from tqdm import tqdm
import math
from functools import partial

from .AI.TAAC import TAAC, Memory
from .env_wrapper import TAACEnvironmentWrapper, create_env_config
from .logger import TAACLogger, extract_environment_metrics, format_time


# --- Constants ---
BASE_SAVE_DIR = "files"
BASE_LOG_DIR = "files/experiments"


def setup_paths(env_name: str, job_name: str) -> Tuple[str, str]:
    """Create unique directories for saving models and logs using standard structure."""
    
    # Standard directory structure: organized by environment, then by job_name
    save_dir = os.path.join("files", "Models", env_name, job_name)
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join("files", "experiments", env_name, job_name)  
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


class PersistentWorker:
    """Persistent worker process that reuses environment for multiple episodes with dynamic agent support"""
    
    def __init__(self, worker_id: int, env_name: str, env_kwargs: Dict, max_steps: int, 
                 env_config: Dict, training_config: Dict, gpu_id: Optional[int] = None, 
                 dynamic_config: Optional[Dict] = None):
        self.worker_id = worker_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs.copy()
        self.max_steps = max_steps
        self.gpu_id = gpu_id
        self.env_config = env_config
        self.training_config = training_config
        self.dynamic_config = dynamic_config or {}
        self.env_wrapper = None
        self.device = None
        self.model = None
        self.max_agents = None  # Track maximum number of agents for memory allocation
        
        # Initialize environment, device, and model
        self._setup_worker()
    
    def _setup_worker(self):
        """Setup environment, device, and model for this worker"""
        print(f"Initializing Worker {self.worker_id} (PID: {os.getpid()}, GPU: {self.gpu_id})")
        
        # Initialize pygame for this process if rendering is enabled
        if self.env_kwargs.get('render_mode') == 'human':
            try:
                pygame.init()
                pygame.display.set_caption(f"TAAC Training - Worker {self.worker_id}")
            except Exception as e:
                print(f"Warning: Could not initialize pygame in worker {self.worker_id}: {e}")
                # Disable rendering for this process if pygame fails
                self.env_kwargs['render_mode'] = None
        
        # Set device for this process
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() and self.gpu_id is not None else "cpu")
        
        # Determine maximum number of agents for memory allocation
        if self.dynamic_config.get('enabled', False):
            agent_counts = self.dynamic_config.get('agent_counts', [])
            self.max_agents = max(agent_counts) if agent_counts else self.env_config['num_agents']
            print(f"🔧 Dynamic Agent Config: Agent counts {agent_counts}, Max agents: {self.max_agents}")
        else:
            self.max_agents = self.env_config['num_agents']
        
        # Create environment for this worker
        self.env_wrapper = TAACEnvironmentWrapper(
            self.env_name, 
            dynamic_config=self.dynamic_config, 
            **self.env_kwargs
        )
        
        # Use configuration with maximum number of agents for model creation
        model_env_config = self.env_config.copy()
        model_env_config['num_agents'] = self.max_agents
        
        # Create model for this worker
        self.model = TAAC(model_env_config, self.training_config, mode="train")
        if hasattr(self.model, 'assign_device'):
            self.model.assign_device(self.device)
        
        print(f"Worker {self.worker_id} initialized successfully on {self.device}")
    
    def run_episode(self, model_state_dict: Dict, episode_num: int, agent_count: Optional[int] = None, 
                   termination_height: Optional[float] = None) -> Tuple[Tuple[float, Optional[float], Dict], List]:
        """Run a single episode with the given model state, potentially with specified agent count"""
        # Update model with new state dict
        if model_state_dict:
            try:
                self.model.load_state_dict(model_state_dict)
            except Exception as e:
                print(f"Warning: Could not load state dict in worker {self.worker_id}: {e}")
        
        # If agent count is specified, recreate environment with that count
        if agent_count is not None and self.dynamic_config.get('enabled', False):
            # Update environment kwargs with specified agent count
            updated_env_kwargs = self.env_kwargs.copy()
            if self.env_name == 'boxjump':
                updated_env_kwargs['num_boxes'] = agent_count
            else:
                updated_env_kwargs['num_agents'] = agent_count
                
            # Update termination height if specified
            if termination_height is not None:
                updated_env_kwargs['termination_max_height'] = termination_height
                if 'termination_reward' not in updated_env_kwargs:
                    adaptive_config = self.dynamic_config.get('adaptive_termination', {})
                    updated_env_kwargs['termination_reward'] = adaptive_config.get('base_reward', 100.0)
            
            # Close current environment and create new one
            if self.env_wrapper:
                self.env_wrapper.close()
                
            self.env_wrapper = TAACEnvironmentWrapper(self.env_name, **updated_env_kwargs)
            print(f"Worker {self.worker_id}: Episode {episode_num} - Recreated env with {agent_count} agents")
        
        # Reset environment
        states, info = self.env_wrapper.reset()
            
        # Ensure memory is properly prepared for this episode
        current_num_agents = self.env_wrapper.num_agents
        self.model.memory_prep(current_num_agents)
        print(f"Worker {self.worker_id}: num memories = {len(self.model.memories)} for {current_num_agents} agents")
            
        episode_reward = 0
        episode_entropies = []
        step_count = 0
        done = False
            
        # Track environment-specific metrics
        all_states = []
        all_rewards = []
        all_infos = []
            
        print(f"Worker {self.worker_id}: Starting episode {episode_num}")
            
        # Main episode loop
        while not done and step_count < self.max_steps:
            # IMPORTANT: This also stores states and actions in the memory
            train_actions, train_log_probs, train_entropies = self.model.get_actions(states)
                
            # Track entropies for logging
            if step_count % 10 == 0:
                episode_entropies.append(train_entropies)
                
            # Step environment
            next_states, rewards, done, env_info = self.env_wrapper.step(train_actions)
                
            # CRITICAL: Store rewards and done signals
            self.model.store_rewards(rewards, done)
                
            # Track states and rewards for metrics
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
                self.env_wrapper.action_size
            )
            
        # Extract environment-specific metrics
        final_states = all_states[-1] if all_states else []
        final_rewards = all_rewards[-1] if all_rewards else {}
        final_info = all_infos[-1] if all_infos else {}
            
        env_metrics = extract_environment_metrics(
            self.env_name, final_states, final_rewards, final_info, 
            all_states_history=all_states
        )

        return (episode_reward, normalized_entropy, env_metrics), self.model.memories
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.env_wrapper:
                self.env_wrapper.close()
        except:
            pass
        
        # Clean up pygame if it was initialized
        if self.env_kwargs.get('render_mode') == 'human':
            try:
                pygame.display.quit()
                pygame.quit()
            except:
                pass


def worker_process(worker_id: int, env_name: str, env_kwargs: Dict, max_steps: int, 
                  env_config: Dict, training_config: Dict, gpu_id: Optional[int], 
                  task_queue, result_queue, dynamic_config: Optional[Dict] = None):
    """Persistent worker process function"""
    worker = PersistentWorker(worker_id, env_name, env_kwargs, max_steps, 
                             env_config, training_config, gpu_id, dynamic_config)
    
    try:
        while True:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break
            
            # Handle both old format (tuple) and new format (dict)
            if isinstance(task, dict):
                model_state_dict = task['model_state_dict']
                episode_num = task['episode_num']
                agent_count = task.get('agent_count')
                termination_height = task.get('termination_height')
            else:
                # Old format for backward compatibility
                model_state_dict, episode_num = task
                agent_count = None
                termination_height = None
                
            result = worker.run_episode(model_state_dict, episode_num, agent_count, termination_height)
            result_queue.put((worker_id, result))
            
    except Exception as e:
        print(f"Worker {worker_id} crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        worker.cleanup()


def train_taac_parallel(config: Dict[str, Any], num_parallel_games: int = 4) -> TAAC:
    """
    Main training loop for TAAC algorithm (parallel environments with persistent workers)
    Supports dynamic agent training with variable agent counts per episode.
    """
    # Extract configuration
    env_name = config['environment']['name']
    job_name = config['job_name']
    dynamic_config = config.get('dynamic_agents', {})
    
    # Setup paths using standard structure: files/{Models|experiments}/{env_name}/{job_name}/
    save_dir, log_dir = setup_paths(env_name, job_name)
    
    # Save configuration
    save_config(config, log_dir)
    
    print(f"=> Starting TAAC Parallel Training (Persistent Workers)")
    print(f"Environment: {env_name}")
    print(f"Job Name: {job_name}")
    print(f"Model Directory: {save_dir}")
    print(f"Log Directory: {log_dir}")
    print(f"Training episodes: {config['training']['episodes']}")
    print(f"Parallel workers: {num_parallel_games}")
    
    # Print dynamic agent configuration
    if dynamic_config.get('enabled', False):
        agent_counts = dynamic_config.get('agent_counts', [])
        print(f"🎲 Dynamic Agent Training ENABLED:")
        print(f"   Agent counts: {agent_counts}")
        if dynamic_config.get('adaptive_termination', {}).get('enabled', False):
            height_formula = dynamic_config.get('adaptive_termination', {}).get('height_formula', 'num_agents + 0.5')
            print(f"   Adaptive termination: {height_formula}")
    
    # Create sample environment to get info (with dynamic config if enabled)
    sample_env = TAACEnvironmentWrapper(
        env_name, 
        apply_wrappers=config['environment'].get('apply_wrappers', True),
        dynamic_config=dynamic_config,
        **config['environment'].get('env_kwargs', {})
    )
    
    # Environment info
    env_config = sample_env.env_info
    training_config = config['training'].copy()
    
    # For dynamic agent training, use maximum agent count for model initialization
    if dynamic_config.get('enabled', False):
        agent_counts = dynamic_config.get('agent_counts', [])
        max_agents = max(agent_counts) if agent_counts else env_config['num_agents']
        env_config['num_agents'] = max_agents
        print(f"   Model will be sized for maximum {max_agents} agents")
    
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
    stats_update_frequency = config['logging'].get('stats_update_frequency', 100)
    
    # Initialize logger with StatisticsManager
    logger = TAACLogger(env_name, job_name, experiment_dir=log_dir, stats_update_frequency=stats_update_frequency)
    
    # Training variables
    start_time = time.time()
    
    print(f"\n=> Starting persistent worker training for {episodes} episodes...")
    
    # Create multiprocessing context with spawn method for CUDA compatibility
    ctx = mp.get_context('spawn')
    
    # Create task and result queues
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    # Start persistent worker processes
    workers = []
    for i in range(num_parallel_games):
        gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
        worker = ctx.Process(
            target=worker_process,
            args=(i, env_name, config['environment'].get('env_kwargs', {}), 
                  max_steps, env_config, training_config, gpu_id, task_queue, result_queue, dynamic_config)
        )
        worker.start()
        workers.append(worker)
    
    print(f"Started {len(workers)} persistent worker processes")
    
    try:
        # Training loop
        for epoch in tqdm(range(episodes), desc="Training TAAC Parallel"):
            
            # For dynamic agent training, select agent count for this epoch
            current_agent_count = None
            current_termination_height = None
            if dynamic_config.get('enabled', False):
                agent_counts = dynamic_config.get('agent_counts', [])
                current_agent_count = random.choice(agent_counts)
                print(f"\n🎲 Episode {epoch + 1}: Using {current_agent_count} agents")
                
                # Calculate adaptive termination if enabled
                adaptive_config = dynamic_config.get('adaptive_termination', {})
                if adaptive_config.get('enabled', False):
                    height_formula = adaptive_config.get('height_formula', 'num_agents + 0.5')
                    if 'num_agents' in height_formula:
                        formula_parts = height_formula.replace('num_agents', str(current_agent_count))
                        try:
                            current_termination_height = eval(formula_parts)
                        except:
                            current_termination_height = current_agent_count + 0.5
                    else:
                        current_termination_height = float(height_formula)
                    print(f"   Termination height: {current_termination_height}")
            
            # Send tasks to workers (one episode per worker)
            model_state_dict = train_model.state_dict()
            for i in range(num_parallel_games):
                # Include current agent count and termination height in task
                task_data = {
                    'model_state_dict': model_state_dict,
                    'episode_num': epoch,
                    'agent_count': current_agent_count,
                    'termination_height': current_termination_height
                }
                task_queue.put(task_data)
            
            # Collect results from workers
            parallel_rewards = []
            parallel_entropies = []
            parallel_env_metrics = []
            all_worker_memories = []  
            
            for i in range(num_parallel_games):
                worker_id, (result, memory) = result_queue.get()
                episode_reward, normalized_entropy, env_metrics = result
                
                parallel_rewards.append(episode_reward)
                parallel_entropies.append(normalized_entropy)
                parallel_env_metrics.append(env_metrics)
                
                all_worker_memories.append(memory)
            
            all_worker_memories = [item for sublist in all_worker_memories for item in sublist] # flatten the list
            train_model.memories = all_worker_memories
            total_experiences = sum(len(mem.rewards) for mem in train_model.memories)

            similarity_loss = train_model.update()
            
            # Log parallel episode results
            logger.log_parallel_episodes(
                rewards=parallel_rewards,
                entropies=parallel_entropies,
                similarity_losses=[similarity_loss] * len(parallel_rewards),
                env_metrics_list=parallel_env_metrics,
                total_experiences=total_experiences
            )
            
            # Progress reporting
            if (epoch + 1) % log_interval == 0:
                logger.print_progress_report(epoch + 1, episodes, log_interval)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"{job_name}_ep{epoch + 1}.pth")
                train_model.save_model(checkpoint_path)
    
    finally:
        # Shutdown workers
        print("\nShutting down worker processes...")
        for _ in range(num_parallel_games):
            task_queue.put(None)  # Shutdown signal
        
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{job_name}_final.pth")
    train_model.save_model(final_model_path)
    
    # Save final statistics
    logger.save_stats(log_dir, final_model_path=final_model_path)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n=> Parallel Training Complete!")
    print(f"Total Time: {format_time(total_time)}")
    print(f"Final Model: {final_model_path}")
    
    return train_model 