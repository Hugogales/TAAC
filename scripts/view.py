#!/usr/bin/env python3
"""
TAAC Model Viewer Script - Main Entry Point for Model Visualization

This script is the main entry point for viewing trained TAAC models.
It routes to the core viewing functionality in AI/model_viewer.py.

Usage:
    # View model using config's load_model field
    python scripts/view.py --config configs/boxjump.yaml
    
    # View specific model
    python scripts/view.py --config configs/boxjump.yaml --model_path files/Models/boxjump/best_model.pth
    
    # View with custom settings
    python scripts/view.py --config configs/mpe_simple_spread.yaml --episodes 3 --render_delay 0.01
    
    # Non-interactive mode (auto-play)
    python scripts/view.py --config configs/cooking_zoo.yaml --non_interactive
"""

import sys
import os
import argparse
import traceback

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the core viewing functions from the AI module
try:
    from AI.model_viewer import load_config, find_model_path, play_game_ai, replay_game
except ImportError as e:
    print("Error: Could not import model viewer module.")
    print(f"Please ensure that the AI directory and model_viewer.py exist: {e}")
    sys.exit(1)


def main():
    """
    Main entry point for TAAC model viewing script.
    Handles argument parsing and routes to appropriate viewing mode.
    """
    parser = argparse.ArgumentParser(
        description="View trained TAAC models in action.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View model using config's load_model field
  python scripts/view.py --config configs/boxjump.yaml
  
  # View specific model
  python scripts/view.py --config configs/boxjump.yaml --model_path files/Models/boxjump/best_model.pth
  
  # View for specific number of episodes
  python scripts/view.py --config configs/mpe_simple_spread.yaml --episodes 3
  
  # Non-interactive mode with faster playback
  python scripts/view.py --config configs/cooking_zoo.yaml --render_delay 0.01 --non_interactive
  
  # Replay a recorded game (if available)
  python scripts/view.py --replay experiments/boxjump/game_log.json
        """
    )
    
    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=str,
                           help='Path to YAML configuration file (for live AI viewing)')
    mode_group.add_argument('--replay', type=str,
                           help='Path to game log file (for replay mode)')
    
    # Live AI viewing options
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model file (overrides config load_model)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to display (default: 5)')
    parser.add_argument('--render_delay', type=float, default=0.05,
                       help='Delay between steps in seconds (default: 0.05)')
    parser.add_argument('--non_interactive', action='store_true',
                       help='Run without waiting for user input between episodes')
    
    args = parser.parse_args()
    
    try:
        if args.replay:
            # Replay mode
            print(f"=> Starting replay mode...")
            replay_game(args.replay)
            
        else:
            # Live AI viewing mode
            print(f"=> Loading configuration from: {args.config}")
            config = load_config(args.config)
            
            # Find model path
            env_name = config['environment']['name']
            if args.model_path:
                model_path = args.model_path
            else:
                model_path = config.get('load_model', None)
            
            model_path = find_model_path(model_path, env_name)
            print(f"=> Using model: {model_path}")
            
            # Run AI visualization
            play_game_ai(
                config=config,
                model_path=model_path,
                episodes=args.episodes,
                render_delay=args.render_delay,
                interactive=not args.non_interactive
            )
        
        print(f"\n=> Viewing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\nViewing interrupted by user. Exiting gracefully...")
        return 0
        
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 