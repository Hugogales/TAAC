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
if os.path.exists(ENVIRONMENTS_DIR):
    # Ensure the local environments folder has precedence over site-packages
    if ENVIRONMENTS_DIR in sys.path:
        sys.path.remove(ENVIRONMENTS_DIR)
    sys.path.insert(0, ENVIRONMENTS_DIR)
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
    
    elif env_name == 'lbforaging':
        # Level-Based Foraging from https://github.com/semitable/lb-foraging
        # Prefer PettingZoo parallel API if available; otherwise wrap the Gymnasium env
        try:
            try:
                from pettingzoo.contrib.lbforaging import parallel_env as lbf_parallel
                # Map our generic keys to LBF expected kwargs when present
                lbf_kwargs = kwargs.copy()
                if 'num_agents' in lbf_kwargs and 'players' not in lbf_kwargs:
                    lbf_kwargs['players'] = int(lbf_kwargs.pop('num_agents'))
                if 'max_cycles' in lbf_kwargs and 'max_episode_steps' not in lbf_kwargs:
                    lbf_kwargs['max_episode_steps'] = int(lbf_kwargs.pop('max_cycles'))
                # Alias support: allow max_food as shorthand for max_num_food
                if 'max_food' in lbf_kwargs and 'max_num_food' not in lbf_kwargs:
                    lbf_kwargs['max_num_food'] = int(lbf_kwargs.pop('max_food'))
                # Ensure field_size is a tuple (rows, cols) if an int is provided
                if 'field_size' in lbf_kwargs and isinstance(lbf_kwargs['field_size'], int):
                    n = int(lbf_kwargs['field_size'])
                    lbf_kwargs['field_size'] = (n, n)
                return lbf_parallel(**lbf_kwargs)
            except Exception:
                pass

            # Fallback: wrap the Gymnasium ForagingEnv into a PettingZoo-parallel-like API
            from lbforaging.foraging.environment import ForagingEnv  # type: ignore

            lbf_kwargs = kwargs.copy()
            if 'num_agents' in lbf_kwargs and 'players' not in lbf_kwargs:
                lbf_kwargs['players'] = int(lbf_kwargs.pop('num_agents'))
            # Rename common alises
            if 'max_food' in lbf_kwargs and 'max_num_food' not in lbf_kwargs:
                lbf_kwargs['max_num_food'] = int(lbf_kwargs.pop('max_food'))
            if 'max_cycles' in lbf_kwargs and 'max_episode_steps' not in lbf_kwargs:
                lbf_kwargs['max_episode_steps'] = int(lbf_kwargs.pop('max_cycles'))
            # Ensure field_size is a tuple (rows, cols)
            if 'field_size' in lbf_kwargs and isinstance(lbf_kwargs['field_size'], int):
                n = int(lbf_kwargs['field_size'])
                lbf_kwargs['field_size'] = (n, n)
            # Avoid known bug with grid_observation Box shape by disabling if requested
            if lbf_kwargs.get('grid_observation', False):
                print("Warning: lbforaging grid_observation=True is not supported in this viewer; forcing False.")
                lbf_kwargs['grid_observation'] = False

            # Provide required defaults if missing
            lbf_kwargs.setdefault('min_player_level', 1)
            # If max_player_level provided, keep; else default 2
            lbf_kwargs.setdefault('max_player_level', 2)
            lbf_kwargs.setdefault('min_food_level', 1)
            # If user provided max_food_level, pass through; else None
            lbf_kwargs.setdefault('max_food_level', None)
            lbf_kwargs.setdefault('force_coop', True)
            lbf_kwargs.setdefault('normalize_reward', True)
            lbf_kwargs.setdefault('observe_agent_levels', True)
            lbf_kwargs.setdefault('penalty', 0.0)

            # Avoid env-internal render during reset; we'll render after steps
            lbf_kwargs['render_mode'] = None
            base_env = ForagingEnv(**lbf_kwargs)

            class LBForagingParallelWrapper(ParallelEnv):
                def __init__(self, env):
                    self.env = env
                    self.num_players = len(getattr(env, 'players', []))
                    self.possible_agents = [f"agent-{i+1}" for i in range(self.num_players)]
                    self.agents = self.possible_agents[:]
                    self.metadata = getattr(env, 'metadata', {'render_modes': ['human', 'rgb_array'], 'name': "lbforaging_v0"})
                    # Track food spawn statistics per episode
                    self._spawned_food_count = 0
                    self._spawned_food_sum = 0.0

                def reset(self, seed=None, options=None):
                    obs, info = self.env.reset(seed=seed, options=options)
                    self.agents = self.possible_agents[:]
                    # Compute spawned food stats at reset
                    try:
                        field = getattr(self.env, 'field', None)
                        if field is not None:
                            self._spawned_food_count = int(np.count_nonzero(field))
                            self._spawned_food_sum = float(np.sum(field))
                    except Exception:
                        self._spawned_food_count = 0
                        self._spawned_food_sum = 0.0
                    obs_dict = {agent: obs[i] for i, agent in enumerate(self.agents)}
                    # Inject per-episode meta info for metrics
                    info_dict = {
                        agent: {
                            'foods_spawned_count': self._spawned_food_count,
                            'foods_spawned_sum': self._spawned_food_sum,
                        } for agent in self.agents
                    }
                    return obs_dict, info_dict

                def step(self, actions):
                    # actions: dict agent -> discrete action int (0..5). Map to ordered list
                    ordered = [0] * len(self.agents)
                    for i, agent in enumerate(self.agents):
                        if agent in actions:
                            ordered[i] = int(actions[agent])
                    obs, rewards, done, truncated, info = self.env.step(ordered)
                    # Only render when explicitly requested
                    try:
                        rmode = getattr(self.env, 'render_mode', None)
                        if rmode == 'human':
                            self.env.render()
                    except Exception:
                        pass
                    obs_dict = {agent: obs[i] for i, agent in enumerate(self.agents)}
                    # Make rewards fully cooperative: sum across agents then divide by number of agents
                    total_reward = float(np.sum(rewards))
                    avg_reward = total_reward / max(1, len(self.agents))
                    rew_dict = {agent: avg_reward for agent in self.agents}
                    terminations = {agent: bool(done) for agent in self.agents}
                    truncations = {agent: bool(truncated) for agent in self.agents}
                    # Compute remaining food stats
                    try:
                        field = getattr(self.env, 'field', None)
                        if field is not None:
                            remaining_count = int(np.count_nonzero(field))
                            remaining_sum = float(np.sum(field))
                        else:
                            remaining_count = None
                            remaining_sum = None
                    except Exception:
                        remaining_count = None
                        remaining_sum = None
                    # Build info dict with metrics signals
                    success_flag = bool(done and (remaining_count == 0 if remaining_count is not None else False))
                    info_payload = {
                        'foods_spawned_count': self._spawned_food_count,
                        'foods_spawned_sum': self._spawned_food_sum,
                        'foods_remaining_count': remaining_count,
                        'foods_remaining_sum': remaining_sum,
                        'episode_success': success_flag,
                    }
                    info_dict = {agent: dict(info_payload) for agent in self.agents}
                    return obs_dict, rew_dict, terminations, truncations, info_dict

                def observation_space(self, agent):
                    idx = self.possible_agents.index(agent)
                    return self.env.observation_space[idx]

                def action_space(self, agent):
                    idx = self.possible_agents.index(agent)
                    return self.env.action_space[idx]

                def close(self):
                    self.env.close()
                
                def render(self):
                    try:
                        return self.env.render()
                    except Exception:
                        return None

            return LBForagingParallelWrapper(base_env)
        except Exception as e:
            raise ImportError(
                "lbforaging not available. Install with: pip install lbforaging gymnasium pygame\n"
                f"Import error: {e}"
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
        # Intercept wrapper-only parameter: do NOT pass to underlying env
        self.auto_reset_on_termination = bool(env_kwargs.pop('reset_episode_on_termination', False))
        # Keep a seed counter for wrapper-managed resets
        self._autoreset_seed_counter = 0
        self.original_env_kwargs = env_kwargs.copy()
        
        # Handle dynamic agent configuration
        self.current_agent_count = None
        self.current_termination_height = None
        if self._is_dynamic_agent_enabled():
            self._setup_dynamic_agents()
            # Prepare kwargs even in dynamic mode so overrides (agent count/termination) are applied
            env_kwargs = self._prepare_env_kwargs(env_kwargs)
        else:
            # Use static configuration
            env_kwargs = self._prepare_env_kwargs(env_kwargs)
            
        # Create environment
        self.env = make_env(env_name, **env_kwargs)
        self.original_env = self.env
        
        # Store BoxJump-specific termination parameters
        self.termination_max_height = env_kwargs.get('termination_max_height', None)
        self.termination_reward_coef = env_kwargs.get('termination_reward_coef', 0.0)
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
            height_formula = adaptive_config.get('height_formula', 'num_agents')
            # Simple formula evaluation (only supports num_agents + number format)
            if 'num_agents' in height_formula:
                # Extract the addition/subtraction part
                formula_parts = height_formula.replace('num_agents', str(self.current_agent_count))
                try:
                    self.current_termination_height = eval(formula_parts)
                except:
                    # Fallback to simple addition
                    self.current_termination_height = self.current_agent_count 
            else:
                self.current_termination_height = float(height_formula)
        
    def _prepare_env_kwargs(self, env_kwargs: Dict) -> Dict:
        """Prepare environment kwargs with dynamic configuration"""
        # Normalize friendly aliases for BoxJump physics controls
        if self.env_name == 'boxjump':
            if 'physics_steps' in env_kwargs and 'physics_steps_per_action' not in env_kwargs:
                env_kwargs['physics_steps_per_action'] = env_kwargs.pop('physics_steps')
            # Accept possible misspelling 'physics_timestep_multipler' and map to correct name
            if 'physics_timestep_multipler' in env_kwargs and 'physics_timestep_multiplier' not in env_kwargs:
                env_kwargs['physics_timestep_multiplier'] = env_kwargs.pop('physics_timestep_multipler')
        
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
            # Always apply dynamic termination height
            env_kwargs['termination_max_height'] = self.current_termination_height
            # Only apply dynamic base reward if YAML/env kwargs didn't specify one
            adaptive_config = self.dynamic_config.get('adaptive_termination', {})
            if ('termination_reward_coef' not in env_kwargs) and ('base_reward' in adaptive_config):
                env_kwargs['termination_reward_coef'] = adaptive_config['base_reward']
                
        return env_kwargs
    
    def get_current_config(self) -> Dict:
        """Get current dynamic configuration info"""
        return {
            'agent_count': self.current_agent_count,
            'termination_height': self.current_termination_height,
            'termination_reward_coef': self.termination_reward_coef,
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
        # Ensure wrapper-only flag never reaches the env on recreation
        if 'reset_episode_on_termination' in env_kwargs:
            env_kwargs.pop('reset_episode_on_termination', None)
        
        # Close current environment
        self.close()
        
        # Create new environment
        self.env = make_env(self.env_name, **env_kwargs)
        self.original_env = self.env
        
        # Update termination parameters
        self.termination_max_height = env_kwargs.get('termination_max_height', None)
        self.termination_reward_coef = env_kwargs.get('termination_reward_coef', 0.0)
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

        # If lbforaging, enforce fully-cooperative reward at each timestep
        if self.env_name == 'lbforaging':
            try:
                if isinstance(rewards, dict) and rewards:
                    total = float(sum(rewards.values()))
                    avg = total / float(self.num_agents)
                    rewards = {agent: avg for agent in rewards.keys()}
            except Exception:
                pass
        
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
            
            # Note: Do not override rewards here; let the environment drive termination and bonuses
            
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

            # Wrapper-managed auto reset only: do not alter env's termination logic
            episode_done = False
            if isinstance(dones, dict) and dones:
                episode_done = all(dones.values())
            if not episode_done:
                try:
                    current_agents = getattr(self.original_env, 'agents', [])
                    if current_agents is not None and len(current_agents) == 0:
                        episode_done = True
                except Exception:
                    pass

            if self.auto_reset_on_termination and episode_done:
                # Determine if termination was due to reaching the max height target
                reached_top = False
                try:
                    pre_reset_heights = []
                    if isinstance(observations, dict):
                        for _agent, obs in observations.items():
                            candidate = None
                            if isinstance(obs, dict):
                                for v in obs.values():
                                    if isinstance(v, np.ndarray) and v.size > 1:
                                        candidate = v
                                        break
                            elif isinstance(obs, np.ndarray):
                                candidate = obs
                            if candidate is not None:
                                flat = candidate.flatten()
                                if flat.size > 1:
                                    pre_reset_heights.append(float(flat[1]))
                    if pre_reset_heights:
                        pre_max = max(pre_reset_heights)
                        if self.termination_max_height is not None:
                            reached_top = (pre_max >= float(self.termination_max_height) - 1e-6)
                except Exception:
                    reached_top = False

                self._autoreset_seed_counter += 1
                reset_obs, reset_info = self.original_env.reset(seed=self._autoreset_seed_counter)
                # Replace observations with reset observations and clear dones
                observations = reset_obs
                # Create fresh dones dict with all False
                if isinstance(observations, dict):
                    dones = {agent: False for agent in observations.keys()}
                else:
                    dones = {agent: False for agent in self.agents}
                if isinstance(info, dict):
                    info['auto_reset'] = True
                    info['reached_termination_height'] = bool(reached_top)
                    try:
                        if pre_reset_heights:
                            info['pre_reset_max_height'] = float(max(pre_reset_heights))
                    except Exception:
                        pass
                else:
                    info = {'auto_reset': True, 'reached_termination_height': bool(reached_top)}
                    try:
                        if pre_reset_heights:
                            info['pre_reset_max_height'] = float(max(pre_reset_heights))
                    except Exception:
                        pass

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
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                # Combine terminations and truncations into dones
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
            else:
                raise ValueError(f"MPE returned unexpected number of values: {len(result)}")
            
            # Wrapper-managed auto reset for MPE
            if self.auto_reset_on_termination:
                episode_done = False
                if isinstance(dones, dict) and dones:
                    episode_done = all(dones.values())
                if not episode_done:
                    try:
                        if hasattr(self.env, 'agents') and len(self.env.agents) == 0:
                            episode_done = True
                    except Exception:
                        pass
                if episode_done:
                    self._autoreset_seed_counter += 1
                    reset_obs, _ = self.env.reset(seed=self._autoreset_seed_counter)
                    observations = reset_obs
                    if isinstance(observations, dict):
                        dones = {agent: False for agent in observations.keys()}
                    else:
                        dones = {agent: False for agent in self.agents}
                    if isinstance(info, dict):
                        info['auto_reset'] = True
                    else:
                        info = {'auto_reset': True}
            
            return observations, rewards, dones, info
                
        except Exception as e:
            print(f"Error in MPE step: {e}")
            raise
    
    def _step_cooking_zoo(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for CookingZoo environment"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
            else:
                raise ValueError(f"CookingZoo returned unexpected number of values: {len(result)}")
            
            # Wrapper-managed auto reset
            if self.auto_reset_on_termination:
                episode_done = False
                if isinstance(dones, dict) and dones:
                    episode_done = all(dones.values())
                if not episode_done:
                    try:
                        if hasattr(self.env, 'agents') and len(self.env.agents) == 0:
                            episode_done = True
                    except Exception:
                        pass
                if episode_done:
                    self._autoreset_seed_counter += 1
                    reset_obs, _ = self.env.reset(seed=self._autoreset_seed_counter)
                    observations = reset_obs
                    if isinstance(observations, dict):
                        dones = {agent: False for agent in observations.keys()}
                    else:
                        dones = {agent: False for agent in self.agents}
                    if isinstance(info, dict):
                        info['auto_reset'] = True
                    else:
                        info = {'auto_reset': True}
            
            return observations, rewards, dones, info
                
        except Exception as e:
            print(f"Error in CookingZoo step: {e}")
            raise
    
    def _step_mats_gym(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Step method specifically for MATS Gym environment"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
            else:
                raise ValueError(f"MATS Gym returned unexpected number of values: {len(result)}")
            
            # Wrapper-managed auto reset
            if self.auto_reset_on_termination:
                episode_done = False
                if isinstance(dones, dict) and dones:
                    episode_done = all(dones.values())
                if not episode_done:
                    try:
                        if hasattr(self.env, 'agents') and len(self.env.agents) == 0:
                            episode_done = True
                    except Exception:
                        pass
                if episode_done:
                    self._autoreset_seed_counter += 1
                    reset_obs, _ = self.env.reset(seed=self._autoreset_seed_counter)
                    observations = reset_obs
                    if isinstance(observations, dict):
                        dones = {agent: False for agent in observations.keys()}
                    else:
                        dones = {agent: False for agent in self.agents}
                    if isinstance(info, dict):
                        info['auto_reset'] = True
                    else:
                        info = {'auto_reset': True}
            
            return observations, rewards, dones, info
                
        except Exception as e:
            print(f"Error in MATS Gym step: {e}")
            raise
    
    def _step_default(self, env_actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Default step method for unknown environments"""
        try:
            result = self.env.step(env_actions)
            
            if len(result) == 4:
                observations, rewards, dones, info = result
            elif len(result) == 5:
                observations, rewards, terminations, truncations, info = result
                dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) 
                        for agent in observations.keys()}
            else:
                # Try to handle other cases gracefully
                print(f"Warning: Environment returned {len(result)} values, expected 4 or 5")
                if len(result) >= 4:
                    observations, rewards, dones, info = result[:4]
                else:
                    raise ValueError(f"Environment returned too few values: {len(result)}")
            
            # Wrapper-managed auto reset for default path
            if self.auto_reset_on_termination:
                episode_done = False
                if isinstance(dones, dict) and dones:
                    episode_done = all(dones.values())
                if not episode_done:
                    try:
                        if hasattr(self.env, 'agents') and len(self.env.agents) == 0:
                            episode_done = True
                    except Exception:
                        pass
                if episode_done:
                    self._autoreset_seed_counter += 1
                    reset_obs, _ = self.env.reset(seed=self._autoreset_seed_counter)
                    observations = reset_obs
                    if isinstance(observations, dict):
                        dones = {agent: False for agent in observations.keys()}
                    else:
                        dones = {agent: False for agent in self.agents}
                    if isinstance(info, dict):
                        info['auto_reset'] = True
                    else:
                        info = {'auto_reset': True}
            
            return observations, rewards, dones, info
                    
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
            'termination_reward_coef': 100.0  # Final reward given to all agents when max height is reached
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