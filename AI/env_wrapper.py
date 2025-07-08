import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
import importlib
from typing import Dict, Any, List, Tuple


def make_env(env_name: str, **kwargs) -> ParallelEnv:
    """
    Create a PettingZoo environment by name
    
    Args:
        env_name: Environment name (e.g., 'cooking_zoo', 'boxjump', 'mats_gym')
        **kwargs: Additional environment parameters
        
    Returns:
        PettingZoo parallel environment
    """
    if env_name == 'cooking_zoo':
        try:
            from cooking_zoo.environment import parallel_env
            return parallel_env(**kwargs)
        except ImportError:
            raise ImportError("cooking_zoo not installed. Install with: pip install cooking_zoo")
    
    elif env_name == 'boxjump':
        try:
            # Assuming boxjump follows PettingZoo convention
            from boxjump import parallel_env
            return parallel_env(**kwargs)
        except ImportError:
            raise ImportError("boxjump environment not found")
    
    elif env_name == 'mats_gym':
        try:
            from mats_gym import parallel_env
            return parallel_env(**kwargs)
        except ImportError:
            raise ImportError("mats_gym not installed")
    
    elif env_name.startswith('mpe_'):
        # Multi-agent particle environments
        env_type = env_name.replace('mpe_', '')
        try:
            from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_world_comm_v3
            env_map = {
                'simple_spread': simple_spread_v3,
                'simple_tag': simple_tag_v3,
                'simple_world_comm': simple_world_comm_v3
            }
            if env_type in env_map:
                # MPE environments support continuous_actions parameter
                return env_map[env_type].parallel_env(**kwargs)
            else:
                raise ValueError(f"Unknown MPE environment: {env_type}")
        except ImportError:
            raise ImportError("PettingZoo MPE environments not installed")
    
    elif env_name.startswith('atari_'):
        # Atari multi-agent environments
        env_type = env_name.replace('atari_', '')
        try:
            module = importlib.import_module(f"pettingzoo.atari.{env_type}")
            return module.parallel_env(**kwargs)
        except ImportError:
            raise ImportError(f"Atari environment {env_type} not found")
    
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def extract_env_info(env: ParallelEnv) -> Dict[str, Any]:
    """
    Extract environment information needed for TAAC configuration
    
    Args:
        env: PettingZoo parallel environment
        
    Returns:
        Dictionary with environment specifications
    """
    # Reset environment to get initial observations
    observations, _ = env.reset()
    
    # Get agent list
    agents = list(env.agents)
    num_agents = len(agents)
    
    # Get observation and action spaces
    first_agent = agents[0]
    obs_space = env.observation_space(first_agent)
    action_space = env.action_space(first_agent)
    
    # Determine state size
    if hasattr(obs_space, 'shape'):
        if len(obs_space.shape) == 1:
            state_size = obs_space.shape[0]
        else:
            # Flatten multi-dimensional observations
            state_size = np.prod(obs_space.shape)
    else:
        # Handle Box spaces or other types
        state_size = obs_space.n if hasattr(obs_space, 'n') else 1
    
    # Determine action space type and size
    if hasattr(action_space, 'n'):
        # Discrete action space
        action_space_type = 'discrete'
        action_size = action_space.n
    elif hasattr(action_space, 'shape'):
        # Continuous action space
        action_space_type = 'continuous'
        action_size = action_space.shape[0] if action_space.shape else 1
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")
    
    return {
        'num_agents': num_agents,
        'state_size': state_size,
        'action_size': action_size,
        'action_space_type': action_space_type,
        'agents': agents,
        'obs_space': obs_space,
        'action_space': action_space
    }


class TAACEnvironmentWrapper:
    """
    Wrapper to standardize different PettingZoo environments for TAAC
    """
    
    def __init__(self, env_name: str, apply_wrappers: bool = True, **env_kwargs):
        """
        Initialize environment wrapper
        
        Args:
            env_name: Name of the environment
            apply_wrappers: Whether to apply SuperSuit wrappers for standardization
            **env_kwargs: Additional environment parameters
        """
        self.env_name = env_name
        self.env = make_env(env_name, **env_kwargs)
        self.original_env = self.env
        
        # Apply standardization wrappers if requested
        if apply_wrappers:
            self._apply_wrappers()
            
        # Extract environment information
        self.env_info = extract_env_info(self.env)
        self.agents = self.env_info['agents']
        self.num_agents = self.env_info['num_agents']
        
    def _apply_wrappers(self):
        """Apply SuperSuit wrappers for observation and action space standardization"""
        try:
            # Pad observations to ensure consistent sizes across agents
            self.env = pad_observations_v0(self.env)
            
            # Pad action spaces if they're different across agents
            if hasattr(self.env, 'action_space'):
                self.env = pad_action_space_v0(self.env)
                
        except Exception as e:
            print(f"Warning: Could not apply wrappers: {e}")
            
    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment and return states in TAAC format"""
        observations, info = self.env.reset()
        
        # Convert to list format expected by TAAC
        states = []
        for agent in self.agents:
            if agent in observations:
                obs = observations[agent]
                # Flatten if multi-dimensional
                if len(obs.shape) > 1:
                    obs = obs.flatten()
                states.append(obs)
            else:
                # Handle terminated agents
                states.append(np.zeros(self.env_info['state_size']))
                
        return states, info
    
    def step(self, actions: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Step environment with actions from TAAC
        
        Args:
            actions: Dictionary of actions from TAAC
            
        Returns:
            states, rewards, done, info
        """
        # Convert TAAC actions to environment format
        env_actions = {}
        for i, agent in enumerate(self.agents):
            if agent in self.env.agents:  # Only include active agents
                action_key = f"agent_{i}"
                if action_key in actions:
                    action = actions[action_key]
                    # Handle different action types
                    if self.env_info['action_space_type'] == 'discrete':
                        env_actions[agent] = int(action)
                    else:
                        env_actions[agent] = np.array(action)
        
        # Step environment
        observations, rewards, terminations, truncations, info = self.env.step(env_actions)
        
        # Convert to TAAC format
        states = []
        reward_list = []
        
        for i, agent in enumerate(self.agents):
            if agent in observations:
                obs = observations[agent]
                if len(obs.shape) > 1:
                    obs = obs.flatten()
                states.append(obs)
                reward_list.append(rewards.get(agent, 0.0))
            else:
                # Handle terminated agents
                states.append(np.zeros(self.env_info['state_size']))
                reward_list.append(0.0)
        
        # Check if episode is done (all agents terminated or truncated)
        done = all(terminations.values()) or all(truncations.values())
        
        return states, reward_list, done, info
    
    def close(self):
        """Close the environment"""
        self.env.close()


def create_env_config(env_name: str, **env_kwargs) -> Dict[str, Any]:
    """
    Create environment configuration for TAAC
    
    Args:
        env_name: Name of the environment
        **env_kwargs: Additional environment parameters
        
    Returns:
        Environment configuration dictionary
    """
    # Create temporary environment to extract specs
    temp_env = TAACEnvironmentWrapper(env_name, **env_kwargs)
    env_config = temp_env.env_info.copy()
    temp_env.close()
    
    return env_config


# Predefined environment configurations
ENV_CONFIGS = {
    'cooking_zoo': {
        'env_kwargs': {
            'num_agents': 2,
            'recipe_id': 'TomatoSalad',
            'max_steps': 200
        },
        'training_config': {
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'c_entropy': 0.02,  # Higher entropy for exploration in cooperative tasks
            'similarity_loss_coef': 0.2,  # Higher for cooperation
            'K_epochs': 8,
            'episodes': 2000
        }
    },
    
    'boxjump': {
        'env_kwargs': {
            'num_agents': 2,
            'max_cycles': 500
        },
        'training_config': {
            'gamma': 0.995,  # Higher gamma for longer episodes
            'learning_rate': 2e-4,
            'c_entropy': 0.01,
            'similarity_loss_coef': 0.1,
            'K_epochs': 10,
            'episodes': 3000
        }
    },
    
    'mpe_simple_spread': {
        'env_kwargs': {
            'N': 3,
            'local_ratio': 0.5,
            'max_cycles': 25,
        },
        'training_config': {
            'gamma': 0.95,
            'learning_rate': 1e-3,
            'c_entropy': 0.05,
            'similarity_loss_coef': 0.15,
            'K_epochs': 4,
            'episodes': 1000
        }
    }
} 