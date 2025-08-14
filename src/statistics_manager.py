#!/usr/bin/env python3
"""
Statistics Manager for TAAC Training

Handles collection, storage, and visualization of training metrics across different environments.
Automatically saves metrics to JSON and generates smoothed matplotlib graphs.
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Some features may be limited.")


def convert_to_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'cpu') and hasattr(obj, 'detach'):  # PyTorch tensor
        return obj.cpu().detach().numpy().item() if obj.numel() == 1 else obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class StatisticsManager:
    """
    Comprehensive statistics manager for TAAC training metrics.
    
    Features:
    - Saves metrics to JSON files in experiment directory
    - Generates smoothed matplotlib graphs
    - Environment-specific metric tracking
    - Configurable update frequency
    - Automatic graph styling and labeling
    """
    
    def __init__(self, experiment_name: str, experiment_dir: str, 
                 env_name: str, update_frequency: int = 100):
        """
        Initialize statistics manager.
        
        Args:
            experiment_name: Name of the experiment
            experiment_dir: Directory to save statistics (should be files/experiments/{name}/)
            env_name: Environment name for environment-specific metrics
            update_frequency: How often to update graphs (every N episodes)
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.env_name = env_name.lower()
        self.update_frequency = update_frequency
        
        # Ensure experiment directory exists
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Core metrics tracked for all environments
        self.core_metrics = {
            'episode': [],
            'avg_reward': [],
            'avg_entropy': [],
            'avg_sim_loss': [],
            'timestamp': []
        }
        
        # Environment-specific metrics
        self.env_metrics = {}
        self._init_env_specific_metrics()
        
        # All metrics combined
        self.all_metrics = {**self.core_metrics, **self.env_metrics}
        
        # File paths
        self.json_path = os.path.join(experiment_dir, 'training_metrics.json')
        self.graphs_dir = os.path.join(experiment_dir, 'graphs')
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # Graph styling
        try:
            if 'seaborn-v0_8' in plt.style.available:
                plt.style.use('seaborn-v0_8')
            elif 'seaborn' in plt.style.available:
                plt.style.use('seaborn')
            else:
                plt.style.use('default')
        except Exception:
            plt.style.use('default')
        
        # Load existing data if available
        self._load_existing_data()
        
        print(f"Statistics Manager initialized:")
        print(f"  - Experiment: {experiment_name}")
        print(f"  - Environment: {env_name}")
        print(f"  - Directory: {experiment_dir}")
        print(f"  - Update frequency: every {update_frequency} episodes")
        print(f"  - Tracking metrics: {list(self.all_metrics.keys())}")
    
    def _init_env_specific_metrics(self):
        """Initialize environment-specific metrics based on environment name."""
        if self.env_name == "boxjump":
            self.env_metrics = {
                'max_height_achieved': [],
                'stable_agents_ratio': []
            }
        elif self.env_name == "mpe_simple_spread":
            self.env_metrics = {
                'coverage_area': [],
                'collision_count': [],
                'coordination_efficiency': []
            }
        elif self.env_name == "cooking_zoo":
            self.env_metrics = {
                'dishes_completed': [],
                'ingredient_waste': [],
                'cooking_efficiency': []
            }
        elif self.env_name == "mats_gym":
            self.env_metrics = {
                'traffic_flow': [],
                'collision_avoidance': [],
                'route_efficiency': []
            }
        else:
            # Generic environment metrics
            self.env_metrics = {
                'success_rate': [],
                'efficiency_score': [],
                'cooperation_index': []
            }
    
    def _load_existing_data(self):
        """Load existing metrics data if JSON file exists."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                
                # Load data into metrics dictionaries
                for key in self.all_metrics.keys():
                    if key in data:
                        self.all_metrics[key] = data[key]
                        if key in self.core_metrics:
                            self.core_metrics[key] = data[key]
                        elif key in self.env_metrics:
                            self.env_metrics[key] = data[key]
                
                print(f"Loaded {len(self.all_metrics['episode'])} existing episodes from {self.json_path}")
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")
    
    def add_episode_metrics(self, episode: int, avg_reward: float, 
                          avg_entropy: Optional[float] = None,
                          avg_sim_loss: float = 0.0,
                          env_specific_metrics: Optional[Dict[str, Any]] = None):
        """
        Add metrics for a single episode.
        
        Args:
            episode: Episode number
            avg_reward: Average reward across all agents/environments
            avg_entropy: Average entropy (normalized 0-1)
            avg_sim_loss: Average similarity loss
            env_specific_metrics: Dictionary of environment-specific metrics
        """
        # Add core metrics
        self.core_metrics['episode'].append(episode)
        self.core_metrics['avg_reward'].append(avg_reward)
        self.core_metrics['avg_entropy'].append(avg_entropy if avg_entropy is not None else 0.0)
        self.core_metrics['avg_sim_loss'].append(avg_sim_loss)
        self.core_metrics['timestamp'].append(time.time())
        
        # Add environment-specific metrics
        if env_specific_metrics:
            for key in self.env_metrics.keys():
                value = env_specific_metrics.get(key, 0.0)
                self.env_metrics[key].append(value)
        else:
            # Add zeros for missing metrics
            for key in self.env_metrics.keys():
                self.env_metrics[key].append(0.0)
        
        # Update combined metrics
        self.all_metrics = {**self.core_metrics, **self.env_metrics}
        
        # Auto-update graphs and save data
        if episode % self.update_frequency == 0:
            self.save_metrics()
            self.generate_graphs()
            print(f"Statistics updated at episode {episode}")
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        try:
            # Convert all metrics to serializable format
            serializable_metrics = convert_to_serializable(self.all_metrics)
            with open(self.json_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to {self.json_path}: {e}")
    
    def generate_graphs(self, smooth_window: int = 50):
        """
        Generate smoothed matplotlib graphs for all metrics.
        
        Args:
            smooth_window: Window size for moving average smoothing
        """
        if len(self.core_metrics['episode']) < 2:
            print("Not enough data to generate graphs")
            return
        
        try:
            episodes = np.array(self.core_metrics['episode'])
            
            # Generate core metrics graphs
            self._plot_metric(episodes, self.core_metrics['avg_reward'], 
                            'Average Reward', 'Episode', 'Reward', smooth_window, 'reward')
            
            if any(e > 0 for e in self.core_metrics['avg_entropy']):
                self._plot_metric(episodes, self.core_metrics['avg_entropy'], 
                                'Average Entropy (Normalized)', 'Episode', 'Entropy', smooth_window, 'entropy')
            
            self._plot_metric(episodes, self.core_metrics['avg_sim_loss'], 
                            'Average Similarity Loss', 'Episode', 'Similarity Loss', smooth_window, 'sim_loss')
            
            # Generate environment-specific graphs
            for metric_name, values in self.env_metrics.items():
                if len(values) > 0 and any(v != 0 for v in values):
                    title = self._format_metric_title(metric_name)
                    ylabel = self._format_metric_ylabel(metric_name)
                    self._plot_metric(episodes, values, title, 'Episode', ylabel, smooth_window, metric_name)
            
            # Generate combined overview graph
            self._generate_overview_graph(episodes, smooth_window)
            
            print(f"Graphs saved to {self.graphs_dir}")
            
        except Exception as e:
            print(f"Error generating graphs: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metric(self, episodes: np.ndarray, values: List[float], 
                    title: str, xlabel: str, ylabel: str, 
                    smooth_window: int, filename: str):
        """Plot a single metric with smoothing and proper error handling."""
        # Validate input data
        if len(values) < 1:
            print(f"Warning: No data to plot for {title}")
            return
        
        if len(episodes) != len(values):
            print(f"Warning: Episode count ({len(episodes)}) doesn't match value count ({len(values)}) for {title}")
            return
        
        try:
            plt.figure(figsize=(12, 6))
            values = np.array(values)
            
            # Check for valid data
            if not np.any(np.isfinite(values)):
                print(f"Warning: All values are NaN/inf for {title}, skipping plot")
                plt.close()
                return
            
            # Plot raw data (lighter, thicker than before)
            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                plt.plot(episodes[finite_mask], values[finite_mask], alpha=0.4, color='lightblue', linewidth=1.5, label='Raw Data')
            
            # Plot smoothed data (much thicker)
            if len(values) >= smooth_window and np.sum(finite_mask) >= smooth_window:
                smoothed = self._smooth_data(values, smooth_window)
                if len(smoothed) > 0 and np.any(np.isfinite(smoothed)):
                    smooth_finite_mask = np.isfinite(smoothed)
                    plt.plot(episodes[smooth_finite_mask], smoothed[smooth_finite_mask], 
                            color='darkblue', linewidth=4, label=f'Smoothed ({smooth_window}-episode avg)')
            else:
                # If not enough data for smoothing, plot raw data thicker
                if np.any(finite_mask):
                    plt.plot(episodes[finite_mask], values[finite_mask], color='darkblue', linewidth=4, label='Data')
            
            plt.title(f'{title} - {self.experiment_name}', fontsize=14, fontweight='bold')
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text (only for finite values)
            finite_values = values[finite_mask]
            if len(finite_values) > 0:
                stats_text = f'Latest: {finite_values[-1]:.4f}\nMean: {np.mean(finite_values):.4f}\nStd: {np.std(finite_values):.4f}'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save graph
            graph_path = os.path.join(self.graphs_dir, f'{filename}.png')
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for {title}: {e}")
            plt.close()
    
    def _generate_overview_graph(self, episodes: np.ndarray, smooth_window: int):
        """Generate a combined overview graph with multiple subplots."""
        metrics_to_plot = []
        
        # Always include reward
        if len(self.core_metrics['avg_reward']) > 0:
            metrics_to_plot.append(('avg_reward', 'Average Reward', 'Reward'))
        
        # Include entropy if it has non-zero values
        if any(e > 0 for e in self.core_metrics['avg_entropy']):
            metrics_to_plot.append(('avg_entropy', 'Average Entropy', 'Entropy'))
        
        # Include similarity loss
        if len(self.core_metrics['avg_sim_loss']) > 0:
            metrics_to_plot.append(('avg_sim_loss', 'Similarity Loss', 'Loss'))
        
        # Add main environment metric
        if self.env_name == "boxjump" and len(self.env_metrics.get('max_height_achieved', [])) > 0:
            metrics_to_plot.append(('max_height_achieved', 'Maximum Height Achieved', 'Max Height'))
        
        if len(metrics_to_plot) == 0:
            return
        
        # Create subplot layout
        n_plots = len(metrics_to_plot)
        rows = (n_plots + 1) // 2 if n_plots > 1 else 1
        cols = 2 if n_plots > 1 else 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Training Overview - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        for i, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                if metric_key in self.core_metrics:
                    values = np.array(self.core_metrics[metric_key])
                else:
                    values = np.array(self.env_metrics[metric_key])
                
                # Skip empty or invalid data
                if len(values) == 0 or not np.any(np.isfinite(values)):
                    ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(title, fontweight='bold')
                    continue
                
                # Plot only finite values
                finite_mask = np.isfinite(values)
                finite_episodes = episodes[finite_mask]
                finite_values = values[finite_mask]
                
                if len(finite_values) > 0:
                    # Raw data (thicker than before)
                    ax.plot(finite_episodes, finite_values, alpha=0.4, color='lightcoral', linewidth=1.2)
                    
                    # Smoothed data (much thicker)
                    if len(finite_values) >= smooth_window:
                        smoothed = self._smooth_data(finite_values, smooth_window)
                        if len(smoothed) > 0 and np.any(np.isfinite(smoothed)):
                            smooth_finite_mask = np.isfinite(smoothed)
                            ax.plot(finite_episodes[smooth_finite_mask], smoothed[smooth_finite_mask], 
                                   color='darkred', linewidth=3.5)
                    else:
                        # If not enough data for smoothing, plot raw data thicker
                        ax.plot(finite_episodes, finite_values, color='darkred', linewidth=3.5)
                
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error plotting {title} in overview: {e}")
                ax.text(0.5, 0.5, f'Error plotting {title}', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title, fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save overview graph
        overview_path = os.path.join(self.graphs_dir, 'training_overview.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_data(self, data: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to data with proper error handling."""
        # Validate input data
        if len(data) == 0:
            return np.array([])
        
        if len(data) < window:
            return data
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(data)):
            # Replace NaN/inf with mean of finite values
            finite_data = data[np.isfinite(data)]
            if len(finite_data) == 0:
                return np.zeros_like(data)  # All values are NaN/inf
            mean_val = np.mean(finite_data)
            data = np.where(np.isfinite(data), data, mean_val)
        
        if HAS_PANDAS:
            try:
                # Use pandas rolling mean for better edge handling
                df = pd.DataFrame({'values': data})
                smoothed = df['values'].rolling(window=window, min_periods=1).mean()
                return smoothed.values
            except Exception as e:
                print(f"Warning: pandas smoothing failed: {e}, falling back to numpy")
        
        # Fallback: simple moving average using numpy
        try:
            smoothed = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i+1]
                if len(window_data) > 0:
                    smoothed[i] = np.mean(window_data)
                else:
                    smoothed[i] = 0.0
            return smoothed
        except Exception as e:
            print(f"Warning: numpy smoothing failed: {e}, returning original data")
            return data
    
    def _format_metric_title(self, metric_name: str) -> str:
        """Format metric name into a readable title."""
        formatted = metric_name.replace('_', ' ').title()
        
        # Special cases
        replacements = {
            'Avg': 'Average',
            'Sim Loss': 'Similarity Loss',
            'Max Height Achieved': 'Maximum Height Achieved',
            'Stable Agents Ratio': 'Stable Agents Ratio'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _format_metric_ylabel(self, metric_name: str) -> str:
        """Format metric name into a y-axis label."""
        ylabel_map = {
            'max_height_achieved': 'Max Height (boxes)',
            'stable_agents_ratio': 'Ratio (0-1)',
            'coverage_area': 'Area Coverage',
            'collision_count': 'Collisions',
            'coordination_efficiency': 'Efficiency (0-1)',
            'success_rate': 'Success Rate (0-1)',
            'efficiency_score': 'Efficiency Score'
        }
        
        return ylabel_map.get(metric_name, self._format_metric_title(metric_name))
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest recorded metrics."""
        if len(self.core_metrics['episode']) == 0:
            return {}
        
        latest = {}
        for key, values in self.all_metrics.items():
            if len(values) > 0:
                latest[key] = values[-1]
        
        return latest
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        stats = {}
        
        for key, values in self.all_metrics.items():
            if key == 'timestamp' or len(values) == 0:
                continue
                
            values_array = np.array(values)
            stats[key] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'latest': float(values_array[-1]) if len(values_array) > 0 else 0.0
            }
        
        return stats
    
    def export_to_csv(self, filename: Optional[str] = None):
        """Export all metrics to CSV file."""
        if not HAS_PANDAS:
            print("Warning: pandas not available. Cannot export to CSV.")
            return
            
        if filename is None:
            filename = f'{self.experiment_name}_metrics.csv'
        
        csv_path = os.path.join(self.experiment_dir, filename)
        
        try:
            df = pd.DataFrame(self.all_metrics)
            df.to_csv(csv_path, index=False)
            print(f"Metrics exported to {csv_path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
