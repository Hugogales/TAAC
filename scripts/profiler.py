#!/usr/bin/env python3
"""
TAAC Training Profiler - Profile Training Performance for Optimization

This script profiles TAAC training with 1 parallel environment and outputs
a profile file that can be viewed with snakeviz for performance analysis.

Usage:
    python scripts/profile.py --config configs/boxjump.yaml
    python scripts/profile.py --config configs/mpe_simple_spread.yaml --episodes 10
    
After running, view the profile with:
    pip install snakeviz
    snakeviz profiles/training_profile.prof
"""

import argparse
import yaml
import os
import sys
import cProfile
import pstats
import time
from pathlib import Path
from datetime import datetime

# Add AI directory to path
sys.path.append(str(Path(__file__).parent.parent / "AI"))

from train_taac import main_with_args as train_main


def load_config(config_path):
    """Load and validate configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['environment', 'training', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def setup_directories():
    """Create necessary directories for profiling output."""
    profile_dir = Path("profiles")
    profile_dir.mkdir(exist_ok=True)
    return profile_dir


def config_to_args(config, args):
    """Convert config file to command line arguments for train_taac.py."""
    cmd_args = []
    
    # Environment
    env_name = config['environment']['name']
    cmd_args.extend(['--env', env_name])
    
    # Training parameters (optimized for profiling)
    training = config.get('training', {})
    
    # Use provided episodes or default to small number for profiling
    episodes = args.episodes or 5
    cmd_args.extend(['--episodes', str(episodes)])
    
    if 'learning_rate' in training:
        cmd_args.extend(['--learning_rate', str(training['learning_rate'])])
    if 'gamma' in training:
        cmd_args.extend(['--gamma', str(training['gamma'])])
    
    # Logging parameters (reduced for profiling)
    logging = config.get('logging', {})
    log_interval = max(1, episodes // 2)  # Log at least twice
    cmd_args.extend(['--log_interval', str(log_interval)])
    
    # Disable frequent saves during profiling
    cmd_args.extend(['--save_interval', str(episodes * 2)])  # Save at end only
    cmd_args.extend(['--eval_interval', str(episodes * 2)])  # No evaluation during profiling
    
    # Output paths
    output = config.get('output', {})
    if 'model_save_path' in output:
        cmd_args.extend(['--model_save_path', output['model_save_path']])
    
    # Check for load_model in config
    if 'load_model' in config and config['load_model']:
        cmd_args.extend(['--model_path', str(config['load_model'])])
        print(f"=> Loading existing model: {config['load_model']}")
    
    # Force single parallel environment for profiling
    cmd_args.extend(['--num_parallel', '1'])
    
    return cmd_args


def create_mock_args(cmd_args):
    """Create mock args object for train_taac."""
    import argparse as ap
    train_parser = ap.ArgumentParser()
    train_parser.add_argument('--env', default='mpe_simple_spread')
    train_parser.add_argument('--episodes', type=int, default=5)
    train_parser.add_argument('--learning_rate', type=float, default=3e-4)
    train_parser.add_argument('--gamma', type=float, default=0.99)
    train_parser.add_argument('--log_interval', type=int, default=2)
    train_parser.add_argument('--save_interval', type=int, default=100)
    train_parser.add_argument('--eval_interval', type=int, default=100)
    train_parser.add_argument('--model_save_path', default='files/Models/')
    train_parser.add_argument('--eval_only', action='store_true')
    train_parser.add_argument('--model_path', default=None)
    train_parser.add_argument('--render', action='store_true')
    train_parser.add_argument('--num_parallel', type=int, default=1)
    
    train_args = train_parser.parse_args(cmd_args)
    return train_args


def run_profiled_training(config, profile_path, args):
    """Run training under profiler."""
    print(f"=> Starting profiled training...")
    print(f"=> Profile will be saved to: {profile_path}")
    
    # Convert config to args
    cmd_args = config_to_args(config, args)
    train_args = create_mock_args(cmd_args)
    train_args.config = config
    
    print(f"=> Training with args: {' '.join(cmd_args)}")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    start_time = time.time()
    
    try:
        # Run training under profiler
        profiler.enable()
        train_main(train_args)
        profiler.disable()
        
        elapsed_time = time.time() - start_time
        print(f"=> Training completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        profiler.disable()
        print(f"=> Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save profile
    profiler.dump_stats(str(profile_path))
    print(f"=> Profile saved to: {profile_path}")
    
    return profiler


def analyze_profile(profile_path, config):
    """Analyze the profile and print summary."""
    print(f"\n=> Analyzing profile...")
    
    # Load profile stats
    stats = pstats.Stats(str(profile_path))
    
    # Sort by cumulative time
    stats.sort_stats('cumulative')
    
    print(f"\n=== TOP 20 FUNCTIONS BY CUMULATIVE TIME ===")
    stats.print_stats(20)
    
    print(f"\n=== TOP 20 FUNCTIONS BY TOTAL TIME ===")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Environment-specific analysis
    env_name = config['environment']['name']
    print(f"\n=== ENVIRONMENT-SPECIFIC ANALYSIS ({env_name}) ===")
    
    if env_name == 'boxjump':
        print(f"Looking for BoxJump-specific bottlenecks...")
        stats.print_stats('box.*step|box.*reset|BoxJump')
    
    print(f"\n=== TAAC-SPECIFIC ANALYSIS ===")
    stats.print_stats('TAAC|get_actions|update|forward')
    
    print(f"\n=== PYTORCH/ML ANALYSIS ===")
    stats.print_stats('torch.*forward|torch.*backward|nn.*forward')


def print_snakeviz_instructions(profile_path):
    """Print instructions for viewing with snakeviz."""
    print(f"\n{'='*60}")
    print(f"PROFILE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Profile saved to: {profile_path}")
    print(f"")
    print(f"To view the interactive profile with snakeviz:")
    print(f"")
    print(f"1. Install snakeviz (if not already installed):")
    print(f"   pip install snakeviz")
    print(f"")
    print(f"2. View the profile:")
    print(f"   snakeviz {profile_path}")
    print(f"")
    print(f"3. This will open a web browser with an interactive profile viewer")
    print(f"   - Use the 'Icicle' view for hierarchical function calls")
    print(f"   - Use the 'Sunburst' view for a radial visualization")
    print(f"   - Click on functions to zoom in and analyze bottlenecks")
    print(f"")
    print(f"Alternative command line analysis:")
    print(f"   python -m pstats {profile_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='TAAC Training Profiler - Profile Performance for Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile training with default episodes (5)
  python scripts/profile.py --config configs/boxjump.yaml
  
  # Profile with specific episode count
  python scripts/profile.py --config configs/mpe_simple_spread.yaml --episodes 10
  
  # Profile and immediately view with snakeviz
  python scripts/profile.py --config configs/cooking_zoo.yaml --view
  
After profiling, view results with:
  pip install snakeviz
  snakeviz profiles/training_profile_TIMESTAMP.prof
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, 
                       help='Path to YAML configuration file')
    
    # Optional arguments
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to profile (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output profile filename (default: auto-generated)')
    parser.add_argument('--view', action='store_true',
                       help='Automatically open snakeviz after profiling')
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Print profile analysis (default: True)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"=> Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Setup directories
        profile_dir = setup_directories()
        
        # Generate profile filename
        env_name = config['environment']['name']
        episodes = args.episodes or 5
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output:
            profile_filename = args.output
        else:
            profile_filename = f"training_profile_{env_name}_{episodes}ep_{timestamp}.prof"
        
        profile_path = profile_dir / profile_filename
        
        # Show profiling info
        print(f"=> Environment: {env_name}")
        print(f"=> Episodes: {episodes}")
        print(f"=> Parallel environments: 1 (forced for profiling)")
        print(f"=> Profile output: {profile_path}")
        
        # Run profiled training
        profiler = run_profiled_training(config, profile_path, args)
        
        # Analyze profile
        if args.analyze:
            analyze_profile(profile_path, config)
        
        # Print snakeviz instructions
        print_snakeviz_instructions(profile_path)
        
        # Optionally open snakeviz
        if args.view:
            try:
                import subprocess
                print(f"\n=> Opening snakeviz...")
                subprocess.run(['snakeviz', str(profile_path)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"=> Could not open snakeviz automatically")
                print(f"=> Install with: pip install snakeviz")
                print(f"=> Then run: snakeviz {profile_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 