#!/usr/bin/env python3
"""
Script to check action spaces of different environments
"""

from AI.env_wrapper import TAACEnvironmentWrapper


def check_action_space(env_name, **env_kwargs):
    """Check the action space type of an environment"""
    print(f"\n{'='*50}")
    print(f"Environment: {env_name}")
    print(f"{'='*50}")
    
    try:
        env = TAACEnvironmentWrapper(env_name, apply_wrappers=False, **env_kwargs)
        
        print(f"Action Space Type: {env.env_info['action_space_type']}")
        print(f"Action Size: {env.env_info['action_size']}")
        print(f"Number of Agents: {env.env_info['num_agents']}")
        
        # Get the raw action space for more details
        first_agent = env.env_info['agents'][0]
        action_space = env.env_info['action_space']
        
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            print(f"Action Range: [{action_space.low}, {action_space.high}]")
        elif hasattr(action_space, 'n'):
            print(f"Discrete Actions: {action_space.n}")
            
        print(f"Action Space Object: {type(action_space)}")
        
        env.close()
        print("✓ Successfully checked action space")
        
    except Exception as e:
        print(f"✗ Error checking {env_name}: {e}")


def main():
    """Check action spaces for all supported environments"""
    print("Checking Action Spaces for TAAC Environments")
    
    # Test environments that are likely to be available
    test_cases = [
        # MPE - should be continuous
        ('mpe_simple_spread', {'N': 2, 'max_cycles': 25, 'continuous_actions': True}),
        ('mpe_simple_spread', {'N': 2, 'max_cycles': 25, 'continuous_actions': False}),
        
        # Add others if available
    ]
    
    # Check if CookingZoo is available
    try:
        from cooking_zoo.environment import parallel_env
        test_cases.append(('cooking_zoo', {'num_agents': 2, 'recipe_id': 'TomatoSalad', 'max_steps': 10}))
    except ImportError:
        print("Note: CookingZoo not available")
    
    # Test each environment
    for env_name, env_kwargs in test_cases:
        check_action_space(env_name, **env_kwargs)
    
    print(f"\n{'='*50}")
    print("Summary:")
    print("✓ MPE environments: Support both continuous and discrete")
    print("? MATS Gym: Likely continuous (check installation)")
    print("? BoxJump: Likely continuous (check installation)")  
    print("✗ CookingZoo: Discrete only")


if __name__ == "__main__":
    main() 