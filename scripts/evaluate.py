#!/usr/bin/env python3
"""
Environment-specific Evaluation Entrypoint

Routes evaluation to environment-specific evaluators under `src/evaluation/`.
Saves results under `files/evaluation/{environment}/{job_num}`.
"""

import multiprocessing as mp
import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any
import yaml

# Set spawn method for CUDA compatibility before importing torch-dependent modules
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add project root to module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_factory import resolve_model_name  # may be used for metadata in evaluators
from src.evaluation import get_evaluator


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_evaluation_defaults(config: Dict[str, Any]) -> None:
    eval_cfg = config.setdefault('evaluation', {})
    # models: mapping model_name -> model_path
    eval_cfg.setdefault('models', {})
    # metrics to compute/plot
    eval_cfg.setdefault('metrics', ["reward"])  # default to reward only if not specified
    # scalability settings
    eval_cfg.setdefault('episodes', 5)  # episodes per num_agents setting
    eval_cfg.setdefault('max_steps_per_episode', 1000)
    eval_cfg.setdefault('num_agents_list', [])
    # optional target height formula, e.g., "num_agents + 0.1"
    eval_cfg.setdefault('height_formula', None)
    # rendering toggle
    eval_cfg.setdefault('render', False)


def determine_env_name(config: Dict[str, Any]) -> str:
    # Prefer nested environment.name, fallback to legacy top-level environment_name
    if 'environment' in config and isinstance(config['environment'], dict) and 'name' in config['environment']:
        return config['environment']['name']
    env_name = config.get('environment_name')
    if env_name:
        return env_name
    raise KeyError("Environment name not found in config (expected 'environment.name' or 'environment_name')")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with environment-specific metrics."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (single-model override)")
    parser.add_argument("--job_num", type=str, default=None, help="[Ignored] Results now always use job_name from config")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation.episodes (per num_agents)")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--num_agents", type=int, nargs='*', default=None, help="Override evaluation.num_agents_list")

    args = parser.parse_args()

    # Keep third-party warnings quiet
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module="pygame.pkgdata"
    )

    # Load and normalize config
    print(f"=> Loading configuration from: {args.config}")
    config = load_config(args.config)
    ensure_evaluation_defaults(config)

    if args.episodes is not None:
        config['evaluation']['episodes'] = args.episodes
        print(f"=> Overriding evaluation episodes to: {args.episodes}")

    if args.render:
        config['evaluation']['render'] = True
        # Also propagate to environment kwargs for wrappers that expect it
        config.setdefault('environment', {}).setdefault('env_kwargs', {})['render_mode'] = "human"
        print("=> Enabling rendering for evaluation")

    env_name = determine_env_name(config)
    # Determine models to evaluate
    models: Dict[str, str] = dict(config['evaluation'].get('models') or {})
    if args.model_path:
        # Single-model override: use currently-resolved model name from config for convenience
        single_model_name = resolve_model_name(config)
        models = {single_model_name: args.model_path}

    if not models:
        raise ValueError("No models provided. Set evaluation.models in config or pass --model_path.")

    # Always use job_name from config for results directory
    job_name = config.get('job_name')
    if not job_name:
        raise KeyError("Config missing 'job_name' at top-level; required for evaluation output paths.")
    if args.job_num and args.job_num != job_name:
        print(f"Warning: --job_num is ignored. Using job_name from config: {job_name}")
    job_num = str(job_name)
    results_dir = Path("files") / "evaluation" / env_name / job_num
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=> Environment: {env_name}")
    print(f"=> Models: {', '.join([f'{k}' for k in models.keys()])}")
    print(f"=> Results directory: {results_dir}")

    # Route to environment-specific evaluator
    evaluator = get_evaluator(env_name)
    # Allow evaluator to handle multi-model, multi-num_agents evaluation
    evaluator(
        config=config,
        results_dir=str(results_dir),
        models=models,
        metrics=list(config['evaluation'].get('metrics') or []),
        num_agents_list=list(config['evaluation'].get('num_agents_list') or (args.num_agents or [])),
    )

    print("=> Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


