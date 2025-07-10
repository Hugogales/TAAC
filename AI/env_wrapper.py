import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Add environments directory to path for environment imports
ENVIRONMENTS_DIR = Path(__file__).parent.parent / "environments"
if ENVIRONMENTS_DIR.exists() and str(ENVIRONMENTS_DIR) not in sys.path:
    sys.path.append(str(ENVIRONMENTS_DIR))


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
            # Try to import from environments directory first
            try:
                from environments.cooking_zoo.environment import parallel_env
            except ImportError:
                # Fall back to regular import if not found in environments directory
                from cooking_zoo.environment import parallel_env
            return parallel_env(**kwargs)
        except ImportError:
            raise ImportError(
                "cooking_zoo not installed. Install with:\n"
                "cd environments\n"
                "git clone https://github.com/cooking-gym/cooking-gym cooking_zoo\n"
                "cd cooking_zoo && pip install -e ."
            )
    
    elif env_name == 'boxjump':
        try:
            # Try to import from environments directory first
            try:
                from environments.boxjump.box_env import BoxJumpEnvironment
            except ImportError:
                # Fall back to regular import if not found in environments directory
                from boxjump.box_env import BoxJumpEnvironment
            # BoxJump uses custom environment class, need to wrap it
            return _wrap_boxjump_env(**kwargs)
        except ImportError:
            raise ImportError(
                "BoxJump not installed. Install with:\n"
                "cd environments\n"
                "git clone https://github.com/zzbuzzard/boxjump\n"
                "cd boxjump && pip install -e ."
            )
    
    elif env_name == 'mats_gym':
        try:
            # Try to import from environments directory first
            try:
                from environments.mats_gym import parallel_env
            except ImportError:
                # Fall back to regular import if not found in environments directory
                from mats_gym import parallel_env
            return parallel_env(**kwargs)
        except ImportError:
            raise ImportError(
                "mats_gym not installed. Install with:\n"
                "cd environments\n"
                "git clone https://github.com/your-repo/mats_gym\n" 
                "cd mats_gym && pip install -e ."
            )
    
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


def _wrap_boxjump_env(**kwargs) -> ParallelEnv:
    """
    Wrap BoxJump environment to be compatible with PettingZoo parallel API
    
    Args:
        **kwargs: BoxJump environment parameters
        
    Returns:
        PettingZoo-compatible parallel environment
    """
    # Try to import from environments directory first
    try:
        from environments.boxjump.box_env import BoxJumpEnvironment
    except ImportError:
        # Fall back to regular import if not found in environments directory
        from boxjump.box_env import BoxJumpEnvironment
    
    # Convert PettingZoo parameters to BoxJump parameters
    boxjump_kwargs = kwargs.copy()
    
    # Convert max_cycles to max_timestep for BoxJump compatibility
    if 'max_cycles' in boxjump_kwargs:
        boxjump_kwargs['max_timestep'] = boxjump_kwargs.pop('max_cycles')
    
    # Create BoxJump environment
    boxjump_env = BoxJumpEnvironment(**boxjump_kwargs)
    
    # Convert to PettingZoo parallel environment
    # BoxJump should already be compatible, but we may need custom wrapper
    class BoxJumpParallelWrapper:
        def __init__(self, env):
            self.env = env
            # Create agent names in BoxJump format (box-1, box-2, etc.)
            self.agents = [f"box-{i+1}" for i in range(env.num_boxes)]
            self.possible_agents = self.agents[:]
            
        def reset(self, seed=None, options=None):
            # BoxJump reset returns (observations_dict, info_dict) tuple
            result = self.env.reset()
            
            # Handle both tuple and dict return formats
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
                info = {}
            
            # BoxJump returns observations in format: {'box-1': array, 'box-2': array, ...}
            # Map to our agent format: {'box-1': array, 'box-2': array, ...}
            observations = {}
            
            for i, agent in enumerate(self.agents):
                box_name = f"box-{i+1}"  # BoxJump uses 1-indexed box names
                
                # BoxJump directly provides observations by box name
                if box_name in obs:
                    observations[agent] = obs[box_name]
                else:
                    # Fallback: create zero observation
                    observations[agent] = np.zeros(13, dtype=np.float32)
                    
            infos = {agent: {} for agent in self.agents}
            return observations, infos
            
        def step(self, actions):
            # BoxJump expects actions as dict with box names as keys
            # actions input: {'box-1': 0, 'box-2': 1, 'box-3': 2, 'box-4': 3}
            # BoxJump returns 5 values: obs, rewards, terminations, truncations, infos
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Convert to dict format, BoxJump directly provides observations by box name
            observations = {}
            
            for i, agent in enumerate(self.agents):
                box_name = f"box-{i+1}"  # BoxJump uses 1-indexed box names
                
                # BoxJump directly provides observations by box name
                if box_name in obs:
                    observations[agent] = obs[box_name]
                else:
                    # Fallback: create zero observation
                    observations[agent] = np.zeros(13, dtype=np.float32)
            
            # BoxJump already returns dict format, just pass through
            return observations, rewards, terminations, truncations, infos
            
        def observation_space(self, agent):
            # BoxJump observation_space takes an agent argument
            if hasattr(self.env.observation_space, '__call__'):
                # Get agent index for BoxJump (agent format: "box_0", "box_1", etc.)
                agent_idx = int(agent.split('_')[1]) if '_' in agent else 0
                return self.env.observation_space(agent_idx)
            else:
                return self.env.observation_space
            
        def action_space(self, agent):
            # BoxJump action_space takes an agent argument
            if hasattr(self.env.action_space, '__call__'):
                # Get agent index for BoxJump (agent format: "box_0", "box_1", etc.)
                agent_idx = int(agent.split('_')[1]) if '_' in agent else 0
                return self.env.action_space(agent_idx)
            else:
                return self.env.action_space
            
        def close(self):
            self.env.close()
    
    return BoxJumpParallelWrapper(boxjump_env)


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
        
        # Expose common properties as direct attributes for convenience
        self.state_size = self.env_info['state_size']
        self.action_size = self.env_info['action_size'] 
        self.action_space_type = self.env_info['action_space_type']
        
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
                
                # Handle dict observations (from BoxJump)
                if isinstance(obs, dict):
                    # Find the first valid numpy array in the dict
                    valid_obs = None
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray) and value.size > 0:
                            valid_obs = value
                            break
                    if valid_obs is not None:
                        obs = valid_obs
                    else:
                        # Fallback to zero array
                        obs = np.zeros(self.env_info['state_size'], dtype=np.float32)
                
                # Ensure obs is a proper numpy array and flatten if multi-dimensional
                if not isinstance(obs, np.ndarray):
                    # Convert to array if it's not already
                    obs = np.array(obs)
                
                # Ensure we have at least 1D array
                if obs.ndim == 0:
                    obs = obs.reshape(-1)
                elif obs.ndim > 1:
                    obs = obs.flatten()
                    
                # Ensure proper dtype
                if obs.dtype == object:
                    obs = obs.astype(np.float32)
                    
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
                
                # Handle dict observations (from BoxJump)
                if isinstance(obs, dict):
                    # Find the first valid numpy array in the dict
                    valid_obs = None
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray) and value.size > 0:
                            valid_obs = value
                            break
                    if valid_obs is not None:
                        obs = valid_obs
                    else:
                        # Fallback to zero array
                        obs = np.zeros(self.env_info['state_size'], dtype=np.float32)
                
                # Ensure obs is a proper numpy array and flatten if multi-dimensional
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                
                # Ensure we have at least 1D array
                if obs.ndim == 0:
                    obs = obs.reshape(-1)
                elif obs.ndim > 1:
                    obs = obs.flatten()
                    
                # Ensure proper dtype
                if obs.dtype == object:
                    obs = obs.astype(np.float32)
                    
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
            'num_boxes': 4,  # Number of box agents (2-16)
            'fixed_rotation': True,  # Disable rotation for easier coordination
            'render_mode': None,  # None for training, "human" for visualization
            'max_timestep': 500  # BoxJump uses max_timestep, not max_cycles
        },
        'training_config': {
            'gamma': 0.995,  # Higher gamma for delayed tower-building rewards
            'learning_rate': 3e-4,  # Standard learning rate for discrete actions
            'c_entropy': 0.05,  # Higher entropy for exploration of building strategies
            'similarity_loss_coef': 0.3,  # High cooperation for tower building
            'K_epochs': 8,
            'episodes': 2000,
            'batch_size': 64  # Moderate batch size for 4-action discrete space
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