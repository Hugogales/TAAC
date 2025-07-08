#!/usr/bin/env python3
"""
Example usage of TAAC on different environments
This script demonstrates basic usage patterns
"""

import os
import sys
from AI.env_wrapper import TAACEnvironmentWrapper, create_env_config
from AI.TAAC import TAAC


def demo_environment_info(env_name, **env_kwargs):
    """Demonstrate environment information extraction"""
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print(f"{'='*60}")
    
    try:
        # Create environment wrapper
        env = TAACEnvironmentWrapper(env_name, apply_wrappers=True, **env_kwargs)
        
        # Display environment info
        print(f"Number of agents: {env.num_agents}")
        print(f"State size: {env.env_info['state_size']}")
        print(f"Action size: {env.env_info['action_size']}")
        print(f"Action type: {env.env_info['action_space_type']}")
        print(f"Agents: {env.env_info['agents']}")
        
        # Test one step
        print("\nTesting environment interaction...")
        states, info = env.reset()
        print(f"Initial states shape: {[len(s) for s in states]}")
        
        # Create a simple TAAC agent for demonstration
        env_config = env.env_info
        training_config = {
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'episodes': 10  # Just for demo
        }
        
        agent = TAAC(env_config, training_config, mode="test")
        
        # Get actions and step environment
        actions, entropies = agent.get_actions(states)
        print(f"Actions generated: {list(actions.keys())}")
        
        next_states, rewards, done, info = env.step(actions)
        print(f"Rewards: {rewards}")
        print(f"Episode done: {done}")
        
        env.close()
        print("✓ Environment test successful!")
        
    except Exception as e:
        print(f"✗ Error testing {env_name}: {e}")
        print("  Make sure the environment is properly installed")


def demo_config_creation():
    """Demonstrate configuration creation"""
    print(f"\n{'='*60}")
    print("Configuration Creation Demo")
    print(f"{'='*60}")
    
    try:
        # Create config for MPE environment
        env_config = create_env_config('mpe_simple_spread', N=2, max_cycles=25)
        print("MPE Simple Spread Configuration:")
        for key, value in env_config.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"✗ Error creating config: {e}")


def main():
    """Run demonstration examples"""
    print("TAAC Environment-Agnostic Demo")
    print("This script tests different environments and shows basic usage")
    
    # Test available environments
    test_environments = [
        ('mpe_simple_spread', {'N': 2, 'max_cycles': 5}),
    ]
    
    # Add other environments if available
    try:
        from cooking_zoo.environment import parallel_env
        test_environments.append(('cooking_zoo', {'num_agents': 2, 'recipe_id': 'TomatoSalad', 'max_steps': 10}))
    except ImportError:
        print("Note: CookingZoo not available for demo")
    
    # Test each available environment
    for env_name, env_kwargs in test_environments:
        demo_environment_info(env_name, **env_kwargs)
    
    # Demo configuration creation
    demo_config_creation()
    
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run setup: python setup_environment.py")
    print("2. Train on MPE: python AI/train_taac.py --env mpe_simple_spread --episodes 100")
    print("3. Train with config: python AI/train_taac.py --env cooking_zoo --config configs/cooking_zoo.yaml")


if __name__ == "__main__":
    main() 