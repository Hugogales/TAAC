#!/usr/bin/env python3
"""
Logger module for TAAC training with environment-specific metrics tracking.
"""

import numpy as np
import json
import os
import time
import math
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from .statistics_manager import StatisticsManager
except ImportError:
    from statistics_manager import StatisticsManager


class TAACLogger:
    """Enhanced logger for tracking TAAC training metrics across different environments"""
    
    def __init__(self, env_name: str, job_name: Optional[str] = None, 
                 experiment_dir: Optional[str] = None, stats_update_frequency: int = 100):
        self.env_name = env_name
        self.job_name = job_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Core metrics
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.normalized_entropies = []
        self.similarity_losses = []
        
        # Environment-specific metrics
        self.env_specific_metrics = {}
        self._init_env_specific_metrics()
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Initialize Statistics Manager if experiment directory is provided
        self.stats_manager = None
        if experiment_dir:
            try:
                self.stats_manager = StatisticsManager(
                    experiment_name=self.job_name,
                    experiment_dir=experiment_dir,
                    env_name=env_name,
                    update_frequency=stats_update_frequency
                )
                print(f"Statistics Manager initialized for experiment: {self.job_name}")
            except Exception as e:
                print(f"Warning: Could not initialize Statistics Manager: {e}")
                import traceback
                traceback.print_exc()
        
    def _init_env_specific_metrics(self):
        """Initialize environment-specific metric tracking"""
        if self.env_name == "boxjump":
            self.env_specific_metrics = {
                "max_height": [],
            }
        elif self.env_name == "mpe_simple_spread":
            self.env_specific_metrics = {
                "coverage_area": [],
                "collision_count": [],
                "coordination_efficiency": []
            }
        elif self.env_name == "cooking_zoo":
            self.env_specific_metrics = {
                "dishes_completed": [],
                "ingredient_waste": [],
                "cooking_efficiency": []
            }
        elif self.env_name == "mats_gym":
            self.env_specific_metrics = {
                "traffic_flow": [],
                "collision_avoidance": [],
                "route_efficiency": []
            }
        else:
            # Generic environment metrics
            self.env_specific_metrics = {
                "success_rate": [],
                "efficiency_score": []
            }
    
    def log_episode(self, 
                   total_reward: float, 
                   entropy: Optional[float] = None,
                   similarity_loss: Optional[float] = None,
                   env_metrics: Optional[Dict[str, Any]] = None):
        """Log a single episode's results with environment-specific metrics"""
        
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        
        if entropy is not None:
            self.normalized_entropies.append(entropy)
            
        if similarity_loss is not None:
            self.similarity_losses.append(similarity_loss)
            
        # Handle environment-specific metrics
        if env_metrics:
            self._log_env_specific_metrics(env_metrics)
        
        # Print episode summary with environment-specific info
        self._print_episode_summary(total_reward, entropy, similarity_loss, env_metrics)
    
    def _log_env_specific_metrics(self, env_metrics: Dict[str, Any]):
        """Log environment-specific metrics"""
        if self.env_name == "boxjump":
            if "max_height" in env_metrics:
                self.env_specific_metrics["max_height"].append(env_metrics["max_height"])
                
        elif self.env_name == "mpe_simple_spread":
            if "coverage_area" in env_metrics:
                self.env_specific_metrics["coverage_area"].append(env_metrics["coverage_area"])
            if "collision_count" in env_metrics:
                self.env_specific_metrics["collision_count"].append(env_metrics["collision_count"])
            if "coordination_efficiency" in env_metrics:
                self.env_specific_metrics["coordination_efficiency"].append(env_metrics["coordination_efficiency"])
                
        elif self.env_name == "cooking_zoo":
            if "dishes_completed" in env_metrics:
                self.env_specific_metrics["dishes_completed"].append(env_metrics["dishes_completed"])
            if "ingredient_waste" in env_metrics:
                self.env_specific_metrics["ingredient_waste"].append(env_metrics["ingredient_waste"])
            if "cooking_efficiency" in env_metrics:
                self.env_specific_metrics["cooking_efficiency"].append(env_metrics["cooking_efficiency"])
                
        elif self.env_name == "mats_gym":
            if "traffic_flow" in env_metrics:
                self.env_specific_metrics["traffic_flow"].append(env_metrics["traffic_flow"])
            if "collision_avoidance" in env_metrics:
                self.env_specific_metrics["collision_avoidance"].append(env_metrics["collision_avoidance"])
            if "route_efficiency" in env_metrics:
                self.env_specific_metrics["route_efficiency"].append(env_metrics["route_efficiency"])
        else:
            # Generic handling
            for key, value in env_metrics.items():
                if key not in self.env_specific_metrics:
                    self.env_specific_metrics[key] = []
                self.env_specific_metrics[key].append(value)
    
    def _print_episode_summary(self, total_reward: float, 
                            entropy: Optional[float], similarity_loss: Optional[float],
                              env_metrics: Optional[Dict[str, Any]]):
        """Print episode summary with environment-specific information"""
        
        # Base metrics
        base_str = f"> Episode {self.episode_count}: Reward: {total_reward:.2f}"
        
        # Add entropy if available
        if entropy is not None:
            base_str += f", Entropy: {entropy:.3f}"
        
        # Add similarity loss if available
        if similarity_loss is not None:
            base_str += f", Sim Loss: {similarity_loss:.4f}"
            
        print(base_str)

        for i in tqdm(range(1), desc=base_str):
            pass
        
        # Environment-specific metrics display
        if env_metrics:
            env_str = "  "
            if self.env_name == "boxjump":
                if "max_height" in env_metrics:
                    env_str += f"Max Height: {env_metrics['max_height']:.2f}, "
                    
            elif self.env_name == "mpe_simple_spread":
                if "coverage_area" in env_metrics:
                    env_str += f"Coverage: {env_metrics['coverage_area']:.2f}, "
                if "collision_count" in env_metrics:
                    env_str += f"Collisions: {env_metrics['collision_count']}, "
                if "coordination_efficiency" in env_metrics:
                    env_str += f"Coordination: {env_metrics['coordination_efficiency']:.2f}"
                    
            elif self.env_name == "cooking_zoo":
                if "dishes_completed" in env_metrics:
                    env_str += f"Dishes: {env_metrics['dishes_completed']}, "
                if "ingredient_waste" in env_metrics:
                    env_str += f"Waste: {env_metrics['ingredient_waste']:.2f}, "
                if "cooking_efficiency" in env_metrics:
                    env_str += f"Efficiency: {env_metrics['cooking_efficiency']:.2f}"
                    
            elif self.env_name == "mats_gym":
                if "traffic_flow" in env_metrics:
                    env_str += f"Traffic Flow: {env_metrics['traffic_flow']:.2f}, "
                if "collision_avoidance" in env_metrics:
                    env_str += f"Collision Avoid: {env_metrics['collision_avoidance']:.2f}, "
                if "route_efficiency" in env_metrics:
                    env_str += f"Route Eff: {env_metrics['route_efficiency']:.2f}"
            
            if env_str.strip() != "":
                print(env_str.rstrip(", "))

                for i in tqdm(range(1), desc=env_str.rstrip(", ")):
                    pass
    
    def log_parallel_episodes(self, 
                             rewards: List[float], 
                             entropies: List[Optional[float]] = None,
                             similarity_losses: List[Optional[float]] = None,
                             env_metrics_list: List[Optional[Dict[str, Any]]] = None,
                             total_experiences: Optional[int] = None):
        """Log multiple episodes from parallel environments"""
        
        # Calculate averages
        avg_reward = np.mean(rewards)
        avg_entropy = np.mean([e for e in entropies]) 
        avg_similarity_loss = np.mean([s for s in similarity_losses]) 
        
        # Update counters
        self.episode_count += 1
        self.episode_rewards.append(avg_reward)
        self.normalized_entropies.append(avg_entropy)
        self.similarity_losses.append(avg_similarity_loss)
        
        # Handle environment-specific metrics
        if env_metrics_list:
            # Average environment-specific metrics
            avg_env_metrics = {}
            for metrics in env_metrics_list:
                if metrics:
                    for key, value in metrics.items():
                        if key not in avg_env_metrics:
                            avg_env_metrics[key] = []
                        avg_env_metrics[key].append(value)
            
            # Calculate averages for environment metrics
            for key, values in avg_env_metrics.items():
                avg_env_metrics[key] = np.mean(values)
            
            self._log_env_specific_metrics(avg_env_metrics)
        
        # Print parallel episode summary
        self._print_parallel_summary(avg_reward, avg_entropy, avg_similarity_loss, rewards, env_metrics_list, total_experiences)
        
        # Update Statistics Manager
        if self.stats_manager:
            env_specific_stats = {}
            if env_metrics_list:
                # Calculate environment-specific statistics for StatisticsManager
                for metrics in env_metrics_list:
                    if metrics:
                        for key, value in metrics.items():
                            if key not in env_specific_stats:
                                env_specific_stats[key] = []
                            env_specific_stats[key].append(value)
                
                # Average the metrics
                for key, values in env_specific_stats.items():
                    env_specific_stats[key] = np.mean(values)
                
                # Add environment-specific metrics based on environment type
                if self.env_name == "boxjump":
                    max_heights = [m.get("max_height") for m in env_metrics_list] 
                    env_specific_stats["max_height_achieved"] = np.max(max_heights)
                    # Calculate ratio of agents that can jump (stable)
                    stable_count = sum(1 for h in max_heights if h > 2)  # Assuming height > 2 means stable
                    env_specific_stats["stable_agents_ratio"] = stable_count / len(max_heights) 
            
            try:
                self.stats_manager.add_episode_metrics(
                    episode=self.episode_count,
                    avg_reward=avg_reward,
                    avg_entropy=avg_entropy,
                    avg_sim_loss=avg_similarity_loss,
                    env_specific_metrics=env_specific_stats
                )
            except Exception as e:
                print(f"Warning: Could not update statistics: {e}")
    
    def _print_parallel_summary(self, avg_reward: float, 
                               avg_entropy: Optional[float], avg_similarity_loss: Optional[float],
                               individual_rewards: List[float],
                               env_metrics_list: List[Optional[Dict[str, Any]]],
                               total_experiences: Optional[int] = None):
        """Print parallel episode summary"""

        base_str = f"> Parallel Episode {self.episode_count}: Avg Reward: {avg_reward:.2f}, Total Experiences: {total_experiences if total_experiences is not None else 'N/A'}"

        if avg_entropy is not None:
            base_str += f", Avg Entropy: {avg_entropy:.3f}"
        if avg_similarity_loss is not None:
            base_str += f", Avg Sim Loss: {avg_similarity_loss:.4f}"
            
        print(base_str)
        print(f"   Individual Rewards: {[f'{r:.1f}' for r in individual_rewards]}")

        for i in tqdm(range(1), desc=base_str + "\n   Individual Rewards: " + str([f'{r:.1f}' for r in individual_rewards])):
            pass
        
        # Environment-specific parallel summary
        if env_metrics_list and any(m is not None for m in env_metrics_list):
            if self.env_name == "boxjump":
                max_heights = [m.get("max_height", 0) for m in env_metrics_list if m and "max_height" in m]
                if max_heights:
                    print(f"   Individual Max Heights: {[f'{h:.1f}' for h in max_heights]}")
                    for i in tqdm(range(1), desc=f"Individual Max Heights: {[f'{h:.1f}' for h in max_heights]}"):
                        pass
                    
            elif self.env_name == "mpe_simple_spread":
                collisions = [m.get("collision_count", 0) for m in env_metrics_list if m and "collision_count" in m]
                if collisions:
                    print(f"   Individual Collisions: {collisions}")
                    for i in tqdm(range(1), desc=f"Individual Collisions: {collisions}"):
                        pass
                    
            elif self.env_name == "cooking_zoo":
                dishes = [m.get("dishes_completed", 0) for m in env_metrics_list if m and "dishes_completed" in m]
                if dishes:
                    print(f"   Individual Dishes: {dishes}")
                    for i in tqdm(range(1), desc=f"Individual Dishes: {dishes}"):
                        pass

    def get_recent_stats(self, window: int = 10) -> Dict[str, float]:
        """Get statistics for the most recent episodes"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_entropies = self.normalized_entropies[-window:] if self.normalized_entropies else []
        recent_similarities = self.similarity_losses[-window:] if self.similarity_losses else []
        
        stats = {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'std_length': np.std(recent_lengths),
        }
        
        if recent_entropies:
            stats.update({
                'avg_entropy': np.mean(recent_entropies),
                'std_entropy': np.std(recent_entropies),
            })
        
        if recent_similarities:
            stats.update({
                'avg_similarity_loss': np.mean(recent_similarities),
                'std_similarity_loss': np.std(recent_similarities),
            })
            
        # Add environment-specific recent stats
        for metric_name, values in self.env_specific_metrics.items():
            if values:
                recent_values = values[-window:]
                stats[f'avg_{metric_name}'] = np.mean(recent_values)
                stats[f'std_{metric_name}'] = np.std(recent_values)
        
        return stats
    
    def print_progress_report(self, episode: int, total_episodes: int, log_interval: int):
        """Print detailed progress report"""
        stats = self.get_recent_stats(log_interval)
        elapsed_time = time.time() - self.start_time
        
        print(f"\n> Episode {episode}/{total_episodes} Progress Report ({self.env_name.upper()}):")
        print(f"  * Time Elapsed: {elapsed_time:.1f}s")
        print(f"  > Avg Reward (last {min(log_interval, len(self.episode_rewards))}): {stats.get('avg_reward', 0):.2f}")
        print(f"  - Avg Length: {stats.get('avg_length', 0):.1f}")
        
        if 'avg_entropy' in stats:
            print(f"  & Avg Entropy: {stats['avg_entropy']:.3f}")
        if 'avg_similarity_loss' in stats:
            print(f"  ~ Avg Similarity Loss: {stats['avg_similarity_loss']:.4f}")
        if 'avg_elo' in stats:
            print(f"  @ Avg ELO: {stats['avg_elo']:.1f}")
        
        # Environment-specific progress
        if self.env_name == "boxjump":
            if 'avg_max_height' in stats:
                print(f"  # Max Height: {stats['avg_max_height']:.2f}")
        elif self.env_name == "mpe_simple_spread":
            if 'avg_collision_count' in stats:
                print(f"  ! Avg Collisions: {stats['avg_collision_count']:.1f}")
        elif self.env_name == "cooking_zoo":
            if 'avg_dishes_completed' in stats:
                print(f"  $ Avg Dishes: {stats['avg_dishes_completed']:.1f}")
        elif self.env_name == "mats_gym":
            if 'avg_traffic_flow' in stats:
                print(f"  % Avg Traffic Flow: {stats['avg_traffic_flow']:.2f}")
        
        episodes_per_sec = episode / elapsed_time if elapsed_time > 0 else 0
        print(f"  => Training Speed: {episodes_per_sec:.2f} episodes/sec")
    
    def save_stats(self, log_dir: str, final_model_path: str = None):
        """Save all training statistics to JSON file"""
        stats = {
            'environment': self.env_name,
            'job_name': self.job_name,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'normalized_entropies': self.normalized_entropies,
            'similarity_losses': self.similarity_losses,
            'env_specific_metrics': self.env_specific_metrics,
            'total_episodes': self.episode_count,
            'final_model_path': final_model_path,
            'training_time': time.time() - self.start_time
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
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
        
        json_stats = convert_numpy_types(stats)
        
        stats_path = os.path.join(log_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"-> Training statistics saved to {stats_path}")

def extract_environment_metrics(env_name: str, states: List[np.ndarray], 
                               rewards: Dict[str, float], info: Dict[str, Any],
                               all_states_history: Optional[List[List[np.ndarray]]] = None) -> Dict[str, Any]:
    """Extract environment-specific metrics from game state"""
    metrics = {}
    
    if env_name == "boxjump":
        # Extract BoxJump-specific metrics
        if states:
            try:
                # Calculate max height across entire episode if history is available
                if all_states_history:
                    # Get all heights from all timesteps during the episode
                    all_heights = []
                    for timestep_states in all_states_history:
                        heights_at_timestep = [state[1] for state in timestep_states if len(state) > 1]
                        all_heights.extend(heights_at_timestep)
                    
                    if all_heights:
                        metrics["max_height"] = max(all_heights)
                else:
                    # Fallback to current timestep only (for model viewer)
                    heights = [state[1] for state in states if len(state) > 1]
                    if heights:
                        metrics["max_height"] = max(heights)
                
                # Calculate tower stability and cooperation score using current states
                heights = [state[1] for state in states if len(state) > 1]
                if heights:
                    metrics["tower_stability"] = np.std(heights)  # Lower is more stable
                
                # Cooperation score based on reward distribution
                if rewards:
                    reward_values = rewards if isinstance(rewards, list) else list(rewards.values())
                    metrics["cooperation_score"] = 1.0 - np.std(reward_values) if len(reward_values) > 1 else 1.0
                
            except (IndexError, TypeError):
                pass
                
    elif env_name == "mpe_simple_spread":
        # Extract MPE-specific metrics
        if states:
            try:
                # Calculate coverage area (spread of agent positions)
                positions = [(state[0], state[1]) for state in states if len(state) >= 2]
                if len(positions) > 1:
                    pos_array = np.array(positions)
                    metrics["coverage_area"] = np.std(pos_array)
                
                # Collision detection (proximity threshold)
                collision_count = 0
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                        if dist < 0.1:  # Collision threshold
                            collision_count += 1
                metrics["collision_count"] = collision_count
                
                # Coordination efficiency (reward variance)
                if rewards:
                    reward_values = rewards if isinstance(rewards, list) else list(rewards.values())
                    metrics["coordination_efficiency"] = np.mean(reward_values)
            except (IndexError, TypeError):
                pass
                
    elif env_name == "cooking_zoo":
        # Extract CookingZoo-specific metrics
        if info:
            metrics["dishes_completed"] = info.get("dishes_completed", 0)
            metrics["ingredient_waste"] = info.get("ingredient_waste", 0.0)
            
        if rewards:
            reward_values = rewards if isinstance(rewards, list) else list(rewards.values())
            metrics["cooking_efficiency"] = np.mean(reward_values)
            
    elif env_name == "mats_gym":
        # Extract MATS Gym-specific metrics
        if info:
            metrics["traffic_flow"] = info.get("traffic_flow", 0.0)
            metrics["collision_avoidance"] = info.get("collision_avoidance", 1.0)
            
        if states:
            try:
                # Route efficiency based on agent movement
                velocities = [np.linalg.norm(state[2:4]) for state in states if len(state) >= 4]
                if velocities:
                    metrics["route_efficiency"] = np.mean(velocities)
            except (IndexError, TypeError):
                pass
    
    return metrics


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" 