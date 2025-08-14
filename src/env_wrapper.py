import os
import sys
import importlib
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0
from pathlib import Path
from pettingzoo.utils.env import ParallelEnv
from supersuit.utils.base_aec_wrapper import BaseWrapper


# Add the environments directory to Python path to ensure proper imports
ENVIRONMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "environments")
if os.path.exists(ENVIRONMENTS_DIR) and ENVIRONMENTS_DIR not in sys.path:
    sys.path.append(ENVIRONMENTS_DIR)
    print(f"Added environments directory to path: {ENVIRONMENTS_DIR}")


def make_env(env_name: str, **kwargs) -> ParallelEnv:
    """
    Create environment by name with appropriate wrappers.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        PettingZoo parallel environment
    """
    if env_name == 'cooking_zoo':
        try:
            # Try to import from environments directory first
            try:
                from environments.cooking_zoo import parallel_env
            except ImportError:
                # Fall back to regular import if not found in environments directory
                from cooking_zoo import parallel_env
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
                # Ensure environments directory is in path
                if ENVIRONMENTS_DIR not in sys.path:
                    sys.path.append(ENVIRONMENTS_DIR)
                    
                # First try direct import from environments subdirectory
                sys.path.append(os.path.join(ENVIRONMENTS_DIR, "boxjump"))
                from box_env import BoxJumpEnvironment
                print("Found BoxJump in environments/boxjump directory")
            except ImportError:
                try:
                    # Try with explicit path
                    from environments.boxjump.box_env import BoxJumpEnvironment
                except ImportError:
                    # Fall back to regular import if not found in environments directory
                    from boxjump.box_env import BoxJumpEnvironment
            
            # BoxJump uses custom environment class, need to wrap it
            return _wrap_boxjump_env(**kwargs)
        except ImportError as e:
            print(f"BoxJump import error details: {str(e)}")
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
                # Force discrete actions for MPE environments
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
        # Try with explicit environments path
        if ENVIRONMENTS_DIR not in sys.path:
            sys.path.append(ENVIRONMENTS_DIR)
            
        # First try direct import from environments subdirectory
        sys.path.append(os.path.join(ENVIRONMENTS_DIR, "boxjump"))
        from box_env import BoxJumpEnvironment
    except ImportError:
        try:
            # Try with module path
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
    # This wrapper is needed because BoxJump doesn't follow the PettingZoo Parallel API precisely.
    # It correctly returns a dict for observations, rewards, etc., but its `step` method
    # returns 5 values instead of 4, and it lacks some metadata attributes.
    class BoxJumpParallelWrapper(ParallelEnv):
        def __init__(self, env):
            self.env = env
            # PettingZoo API requirements
            self.possible_agents = [f"box-{i+1}" for i in range(env.num_boxes)]
            self.agents = self.possible_agents[:]
            self.metadata = getattr(env, 'metadata', {'render_modes': ['human'], 'name': "boxjump_v0"})

        def reset(self, seed=None, options=None):
            # BoxJump reset returns (observations_dict, info_dict) tuple
            result = self.env.reset(seed=seed, options=options)
            if isinstance(result, tuple) and len(result) == 2:
                observations_dict, info_dict = result
                self.agents = list(observations_dict.keys())
                return observations_dict, info_dict
            else:
                # Handle cases where reset might not return info dict
                observations_dict = result
                self.agents = list(observations_dict.keys())
                return observations_dict, {}

        def step(self, actions):
            # BoxJump expects actions as dict with box names as keys
            # actions input: {'box-1': 0, 'box-2': 1, 'box-3': 2, 'box-4': 3}
            # BoxJump returns 5 values: obs, rewards, terminations, truncations, infos
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Combine terminations and truncations into a single `dones` dict
            dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in self.agents}
            
            # The PettingZoo API expects 4 return values, so we pass back dones instead of terminations/truncations
            return obs, rewards, dones, infos
            
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
    if not isinstance(env, (ParallelEnv, BaseWrapper)):
        raise TypeError(f"Expected a PettingZoo ParallelEnv, but got {type(env)}")
        
    # Get a sample agent
    sample_agent = env.possible_agents[0]
    
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
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}. Only discrete action spaces are supported.")
    
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
    Wrapper for environments to be compatible with TAAC training.
    Handles dynamic agent configuration for variable agent count training.
    """
    
    def __init__(self, env_name: str, apply_wrappers: bool = True, dynamic_config: Optional[Dict] = None, **env_kwargs):
        """
        Initialize the environment wrapper
        
        Args:
            env_name: Name of the environment
            apply_wrappers: Whether to apply PettingZoo wrappers
            dynamic_config: Configuration for dynamic agent training
            **env_kwargs: Environment-specific parameters
        """
        self.env_name = env_name
        self.dynamic_config = dynamic_config or {}
        self.original_env_kwargs = env_kwargs.copy()
        
        # Handle dynamic agent configuration
        self.current_agent_count = None
        self.current_termination_height = None
        if self._is_dynamic_agent_enabled():
            self._setup_dynamic_agents()
        else:
            # Use static configuration
            env_kwargs = self._prepare_env_kwargs(env_kwargs)
            
        # Create environment
        self.env = make_env(env_name, **env_kwargs)
        self.original_env = self.env
        
        # Store BoxJump-specific termination parameters
        self.termination_max_height = env_kwargs.get('termination_max_height', None)
        self.termination_reward = env_kwargs.get('termination_reward', 0.0)
        self._episode_terminated = False  # Track if episode has been terminated early
        
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
        
    def _is_dynamic_agent_enabled(self) -> bool:
        """Check if dynamic agent training is enabled"""
        return (self.dynamic_config.get('enabled', False) and 
                'agent_counts' in self.dynamic_config and 
                len(self.dynamic_config['agent_counts']) > 0)
    
    def _setup_dynamic_agents(self):
        """Setup dynamic agent configuration"""
        # Randomly select number of agents for this environment instance
        agent_counts = self.dynamic_config['agent_counts']
        self.current_agent_count = random.choice(agent_counts)
        
        # Calculate adaptive termination height if enabled
        adaptive_config = self.dynamic_config.get('adaptive_termination', {})
        if adaptive_config.get('enabled', False):
            height_formula = adaptive_config.get('height_formula', 'num_agents + 0.5')
            # Simple formula evaluation (only supports num_agents + number format)
            if 'num_agents' in height_formula:
                # Extract the addition/subtraction part
                formula_parts = height_formula.replace('num_agents', str(self.current_agent_count))
                try:
                    self.current_termination_height = eval(formula_parts)
                except:
                    # Fallback to simple addition
                    self.current_termination_height = self.current_agent_count + 0.5
            else:
                self.current_termination_height = float(height_formula)
        
    def _prepare_env_kwargs(self, env_kwargs: Dict) -> Dict:
        """Prepare environment kwargs with dynamic configuration"""
        if not self._is_dynamic_agent_enabled():
            return env_kwargs
            
        # Override environment parameters for dynamic agent training
        if self.dynamic_config.get('override_num_agents', True):
            if self.env_name == 'boxjump':
                env_kwargs['num_boxes'] = self.current_agent_count
            else:
                env_kwargs['num_agents'] = self.current_agent_count
                
        if (self.dynamic_config.get('override_termination', True) and 
            self.current_termination_height is not None):
            env_kwargs['termination_max_height'] = self.current_termination_height
            # Also set termination reward if specified
            adaptive_config = self.dynamic_config.get('adaptive_termination', {})
            if 'base_reward' in adaptive_config:
                env_kwargs['termination_reward'] = adaptive_config['base_reward']
                
        return env_kwargs
    
    def get_current_config(self) -> Dict:
        """Get current dynamic configuration info"""
        return {
            'agent_count': self.current_agent_count,
            'termination_height': self.current_termination_height,
            'termination_reward': self.termination_reward,
            'dynamic_enabled': self._is_dynamic_agent_enabled()
        }
    
    def reset_with_new_agents(self):
        """Reset the environment with a new random agent count"""
        if not self._is_dynamic_agent_enabled():
            return self.reset()
            
        # Setup new dynamic configuration
        self._setup_dynamic_agents()
        
        # Recreate environment with new configuration
        env_kwargs = self._prepare_env_kwargs(self.original_env_kwargs.copy())
        
        # Close current environment
        self.close()
        
        # Create new environment
        self.env = make_env(self.env_name, **env_kwargs)
        self.original_env = self.env
        
        # Update termination parameters
        self.termination_max_height = env_kwargs.get('termination_max_height', None)
        self.termination_reward = env_kwargs.get('termination_reward', 0.0)
        self._episode_terminated = False
        
        # Re-extract environment information
        self.env_info = extract_env_info(self.env)
        self.agents = self.env_info['agents']
        self.num_agents = self.env_info['num_agents']
        
        # Return reset state
        return self.reset()
        
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
        # Reset termination flag
        self._episode_terminated = False
        
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
        # If episode has already been terminated, return terminal state immediately
        if self._episode_terminated:
            # Return terminal state without any further computation
            states = [np.zeros(self.env_info['state_size']) for _ in range(self.num_agents)]
            rewards = [0.0] * self.num_agents
            return states, rewards, True, {'early_termination': True, 'already_terminated': True}
        
        # Convert TAAC actions to environment format
        env_actions = {}
        for i, agent in enumerate(self.agents):
            if agent in self.env.agents:  # Only include active agents
                action_key = f"agent_{i}"
                if action_key in actions:
                    action = actions[action_key]
                    # Convert to int for discrete actions
                    env_actions[agent] = int(action)
        
        # Environment-specific step handling to ensure 4-value returns
        if self.env_name == 'boxjump':
            observations, rewards, dones, info = self._step_boxjump(env_actions)
        elif self.env_name.startswith('mpe_'):
            observations, rewards, dones, info = self._step_mpe(env_actions)
        elif self.env_name == 'cooking_zoo':
            observations, rewards, dones, info = self._step_cooking_zoo(env_actions)
        elif self.env_name == 'mats_gym':
            observations, rewards, dones, info = self._step_mats_gym(env_actions)
        else:
            # Default handling for unknown environments
            observations, rewards, dones, info = self._step_default(env_actions)
        
        # Convert to TAAC format
        states = []
        reward_list = []
        
        for i, agent in enumerate(self.agents):
            if agent in observations:
                obs = observations[agent]
                
                # Handle dict observations (from BoxJump)
                if isinstance(obs, dict):
                    # Convert dict to array if needed
                    if 'observation' in obs:
                        obs = obs['observation']
                    else:
                        # Flatten dict values
                        obs = np.concatenate([np.atleast_1d(v) for v in obs.values()])
                
                # Ensure observation is numpy array
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)
                    
                states.append(obs)
            else:
                # Handle terminated agents
                states.append(np.zeros(self.env_info['state_size']))
                
        # Handle rewards - convert to list in agent order for TAAC compatibility
        for i, agent in enumerate(self.agents):
            if agent in rewards:
                reward_list.append(float(rewards[agent]))
            else:
                reward_list.append(0.0)
        
        # Check if episode is done (all agents done or any agent done depending on environment)
        if isinstance(dones, dict):
            # For most environments, episode is done when all agents are done
            done = all(dones.values()) if dones else False
        else:
            # Handle boolean done
            done = bool(dones)
        
        return states, reward_list, done, info
    
    def _step_boxjump(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for BoxJump environment"""
        try:
            # Access the actual BoxJump environment through the wrapper
            actual_boxjump_env = self.original_env.env if hasattr(self.original_env, 'env') else self.original_env
            
            # Check for early termination BEFORE stepping the environment
            # This prevents unnecessary computation after success
            if (self.termination_max_height is not None and 
                hasattr(actual_boxjump_env, 'highest_y') and 
                actual_boxjump_env.highest_y >= self.termination_max_height):
                
                # Episode should have already terminated - return terminal state
                # Get current observations without stepping
                current_obs = actual_boxjump_env.get_all_obs()
                
                # Create terminal rewards and dones
                rewards = {}
                dones = {}
                for agent in current_obs.keys():
                    dones[agent] = True
                    rewards[agent] = self.termination_reward
                
                # Set termination flag to prevent further steps
                self._episode_terminated = True
                
                # Create terminal info
                info = {
                    'termination_reason': 'max_height_reached',
                    'final_height': actual_boxjump_env.highest_y,
                    'early_termination': True
                }
                
                return current_obs, rewards, dones, info
            
            # Normal environment step
            result = self.original_env.step(env_actions)
            
            if len(result) == 4:
                # BoxJump returns 4 values: observations, rewards, dones, info
                observations, rewards, dones, info = result
            elif len(result) == 5:
                # Handle 5-value return by combining terminations and truncations
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
            else:
                raise ValueError(f"BoxJump returned unexpected number of values: {len(result)}")
            
            # Check for early termination due to max height reached AFTER this step
            if (self.termination_max_height is not None and 
                hasattr(actual_boxjump_env, 'highest_y') and 
                actual_boxjump_env.highest_y >= self.termination_max_height):
                
                # Set all agents as done (terminate episode)
                for agent in observations.keys():
                    dones[agent] = True
                
                # Give termination reward to all agents (override regular rewards)
                if self.termination_reward != 0.0:
                    for agent in observations.keys():
                        rewards[agent] = self.termination_reward
                
                # Set termination flag to prevent further steps
                self._episode_terminated = True
                
                # Add termination info
                info['termination_reason'] = 'max_height_reached'
                info['final_height'] = actual_boxjump_env.highest_y
                info['early_termination'] = True
                
            return observations, rewards, dones, info
                
        except Exception as e:
            print(f"Error in BoxJump step, falling back to wrapped environment: {e}")
            # Fallback to wrapped environment
            return self._step_default(env_actions)
    
    def _step_mpe(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for MPE environments"""
        try:
            # MPE environments typically return 5 values with newer PettingZoo
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
                return observations, rewards, dones, info
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                # Combine terminations and truncations into dones
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
                return observations, rewards, dones, info
            else:
                raise ValueError(f"MPE returned unexpected number of values: {len(result)}")
                
        except Exception as e:
            print(f"Error in MPE step: {e}")
            raise
    
    def _step_cooking_zoo(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for CookingZoo environment"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
                return observations, rewards, dones, info
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
                return observations, rewards, dones, info
            else:
                raise ValueError(f"CookingZoo returned unexpected number of values: {len(result)}")
                
        except Exception as e:
            print(f"Error in CookingZoo step: {e}")
            raise
    
    def _step_mats_gym(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for MATS Gym environment"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
                return observations, rewards, dones, info
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
                return observations, rewards, dones, info
            else:
                raise ValueError(f"MATS Gym returned unexpected number of values: {len(result)}")
                
        except Exception as e:
            print(f"Error in MATS Gym step: {e}")
            raise
    
    def _step_default(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Default step method for unknown environments"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
                return observations, rewards, dones, info
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
                return observations, rewards, dones, info
            else:
                # Try to handle other cases gracefully
                print(f"Warning: Environment returned {len(result)} values, expected 4 or 5")
                if len(result) >= 4:
                    observations, rewards, dones, info = result[:4]
                    return observations, rewards, dones, info
                else:
                    raise ValueError(f"Environment returned too few values: {len(result)}")
                    
        except Exception as e:
            print(f"Error in default step method: {e}")
            raise
    
    def close(self):
        """Close the environment"""
        self.env.close()


def create_env_config(env_name: str, dynamic_config: Optional[Dict] = None, **env_kwargs) -> Dict[str, Any]:
    """
    Create environment configuration for TAAC with dynamic agent support
    
    Args:
        env_name: Name of the environment
        dynamic_config: Configuration for dynamic agent training
        **env_kwargs: Additional environment parameters
        
    Returns:
        Environment configuration dictionary
    """
    # Create temporary environment to extract specs
    temp_env = TAACEnvironmentWrapper(env_name, dynamic_config=dynamic_config, **env_kwargs)
    env_config = temp_env.env_info.copy()
    
    # Add dynamic configuration info if applicable
    if dynamic_config and dynamic_config.get('enabled', False):
        env_config['dynamic_config'] = temp_env.get_current_config()
    
    temp_env.close()
    
    return env_config


def create_dynamic_environment_wrapper(env_name: str, config: Dict[str, Any]) -> TAACEnvironmentWrapper:
    """
    Create an environment wrapper with dynamic agent configuration
    
    Args:
        env_name: Name of the environment
        config: Full configuration dictionary including dynamic_agents section
        
    Returns:
        TAACEnvironmentWrapper instance with dynamic configuration
    """
    # Extract dynamic configuration
    dynamic_config = config.get('dynamic_agents', {})
    
    # Extract environment kwargs
    env_kwargs = config.get('environment', {}).get('env_kwargs', {})
    apply_wrappers = config.get('environment', {}).get('apply_wrappers', True)
    
    # Create wrapper with dynamic configuration
    return TAACEnvironmentWrapper(
        env_name=env_name,
        apply_wrappers=apply_wrappers,
        dynamic_config=dynamic_config,
        **env_kwargs
    )


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
            'max_timestep': 500,  # BoxJump uses max_timestep, not max_cycles
            'termination_max_height': 10.0,  # Terminate episode when this height is reached
            'termination_reward': 100.0  # Final reward given to all agents when max height is reached
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