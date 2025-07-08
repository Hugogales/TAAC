#!/usr/bin/env python3
"""
Setup script for TAAC multi-environment testing
This script helps install dependencies and test environment compatibility
"""

import subprocess
import sys
import importlib
import os


def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} is available")
        return True
    except ImportError:
        print(f"✗ {module_name} is not available")
        if package_name:
            print(f"  Install with: pip install {package_name}")
        return False


def test_basic_imports():
    """Test basic required imports"""
    print("Testing basic dependencies...")
    
    required_modules = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("pettingzoo", "pettingzoo"),
        ("supersuit", "supersuit"),
        ("yaml", "pyyaml"),
        ("matplotlib", "matplotlib")
    ]
    
    all_available = True
    for module, package in required_modules:
        if not check_import(module, package):
            all_available = False
    
    return all_available


def test_environment_availability():
    """Test which environments are available"""
    print("\nTesting environment availability...")
    
    # Test PettingZoo MPE
    try:
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env()
        env.reset()
        env.close()
        print("✓ PettingZoo MPE environments available")
    except Exception as e:
        print(f"✗ PettingZoo MPE environments not available: {e}")
    
    # Test CookingZoo
    try:
        from cooking_zoo.environment import parallel_env
        env = parallel_env(num_agents=2, recipe_id="TomatoSalad", max_steps=10)
        env.reset()
        env.close()
        print("✓ CookingZoo environment available")
    except ImportError:
        print("✗ CookingZoo not available")
        print("  Install with: pip install cooking_zoo")
    except Exception as e:
        print(f"✗ CookingZoo error: {e}")
    
    # Test other environments
    environments_to_test = [
        ("boxjump", "boxjump"),
        ("mats_gym", "mats_gym")
    ]
    
    for env_name, import_name in environments_to_test:
        try:
            module = importlib.import_module(import_name)
            print(f"✓ {env_name} environment available")
        except ImportError:
            print(f"✗ {env_name} environment not available")
            print(f"  Check installation instructions for {env_name}")


def test_taac_imports():
    """Test TAAC-specific imports"""
    print("\nTesting TAAC imports...")
    
    try:
        from AI.TAAC import TAAC, AttentionActorCriticNetwork
        print("✓ TAAC core classes available")
        
        from AI.env_wrapper import TAACEnvironmentWrapper, make_env
        print("✓ Environment wrapper available")
        
        from AI.train_taac import train_taac
        print("✓ Training script available")
        
        return True
    except ImportError as e:
        print(f"✗ TAAC imports failed: {e}")
        print("  Make sure you're running from the correct directory")
        return False


def test_sample_environment():
    """Test a sample environment to ensure everything works"""
    print("\nTesting sample environment...")
    
    try:
        from AI.env_wrapper import TAACEnvironmentWrapper
        
        # Test with MPE environment (most likely to be available)
        env = TAACEnvironmentWrapper('mpe_simple_spread', N=2, max_cycles=5)
        states, _ = env.reset()
        
        print(f"✓ Sample environment created successfully")
        print(f"  - Number of agents: {env.num_agents}")
        print(f"  - State size: {env.env_info['state_size']}")
        print(f"  - Action size: {env.env_info['action_size']}")
        print(f"  - Action type: {env.env_info['action_space_type']}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Sample environment test failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'files/Models',
        'experiments',
        'configs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main setup function"""
    print("TAAC Environment Setup")
    print("=" * 50)
    
    # Test basic imports
    if not test_basic_imports():
        print("\n❌ Basic dependencies missing. Install with:")
        print("pip install -r requirements.txt")
        return False
    
    # Test environment availability
    test_environment_availability()
    
    # Test TAAC imports
    if not test_taac_imports():
        print("\n❌ TAAC modules not found. Make sure you're in the project root.")
        return False
    
    # Create directories
    create_directories()
    
    # Test sample environment
    if test_sample_environment():
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Train on MPE environment: python AI/train_taac.py --env mpe_simple_spread")
        print("2. Train with custom config: python AI/train_taac.py --env cooking_zoo --config configs/cooking_zoo.yaml")
        print("3. Evaluate model: python AI/train_taac.py --env mpe_simple_spread --eval_only --model_path <path_to_model>")
        return True
    else:
        print("\n❌ Setup incomplete. Check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 