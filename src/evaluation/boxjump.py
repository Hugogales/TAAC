from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
try:
    import torch
except Exception:
    torch = None

from src.env_wrapper import TAACEnvironmentWrapper
from src.model_factory import resolve_model_name, get_model_class


def _prepare_agent(config: Dict[str, Any], env_config: Dict[str, Any], model_name_override: Optional[str] = None):
    training_config = dict(config.get('training', {}) or {})
    # Merge model hyperparameters if present (helps match saved architecture)
    if 'model' in config and isinstance(config['model'], dict):
        training_config.update(config['model'])

    model_name = model_name_override if model_name_override else resolve_model_name(config)
    ModelClass = get_model_class(model_name)
    agent = ModelClass(env_config, training_config, mode="test")
    return agent, model_name


def _run_evaluations_for_model(
    config: Dict[str, Any],
    results_dir: str,
    model_name: str,
    model_path: str,
    num_agents_list: List[int],
    metrics: List[str],
    base_env_kwargs: Dict[str, Any],
    env_name: str,
    num_episodes: int,
    max_steps: int,
    height_formula: Optional[str],
    device_id: Optional[int] = None,
    enable_tqdm: bool = True,
) -> List[Dict[str, Any]]:
    results_path = Path(results_dir)
    all_episode_rows: List[Dict[str, Any]] = []

    # Assign device if requested
    use_cuda = (torch is not None) and torch.cuda.is_available()
    if device_id is not None and use_cuda:
        try:
            torch.cuda.set_device(device_id)
        except Exception:
            pass

    for num_agents in num_agents_list:
        env_kwargs = dict(base_env_kwargs)
        env_kwargs['num_boxes'] = int(num_agents)

        env_wrapper = TAACEnvironmentWrapper(
            env_name,
            apply_wrappers=config.get('environment', {}).get('apply_wrappers', False),
            **env_kwargs
        )

        env_config = env_wrapper.env_info
        agent, _resolved = _prepare_agent(config, env_config, model_name_override=model_name)

        # Put agent on device
        try:
            if device_id is not None and use_cuda:
                agent.assign_device(torch.device(f"cuda:{device_id}"))
            else:
                if torch is not None:
                    agent.assign_device(torch.device("cpu"))
        except Exception:
            pass

        # Load model in test mode
        try:
            ok = agent.load_model(model_path, test=True)
        except Exception as e:
            ok = False
            with open(results_path / "load_errors.log", "a") as f:
                f.write(f"[{model_name}] Failed to load {model_path}: {e}\n")
        if not ok:
            env_wrapper.close()
            continue

        try:
            if hasattr(agent, 'policy') and agent.policy is not None:
                agent.policy.eval()
            if hasattr(agent, 'policy_old') and agent.policy_old is not None:
                agent.policy_old.eval()
        except Exception:
            pass

        iterator = range(num_episodes)
        if enable_tqdm:
            iterator = tqdm(iterator, desc=f"{model_name} | agents={num_agents}", leave=False)

        for ep in iterator:
            states, _ = env_wrapper.reset()
            episode_reward = 0.0
            step_count = 0
            done = False

            max_height = 0.0
            steps_to_height_events: List[int] = []
            target_height = None
            if height_formula:
                try:
                    expression = str(height_formula).replace('num_agents', str(int(num_agents)))
                    target_height = float(eval(expression))
                except Exception:
                    try:
                        target_height = float(height_formula)
                    except Exception:
                        target_height = None

            while not done and step_count < max_steps:
                actions, _, _ = agent.get_actions(states)
                states, rewards, done, info = env_wrapper.step(actions)

                episode_reward += float(np.sum(rewards))

                current_best = None
                try:
                    underlying = env_wrapper.original_env.env if hasattr(env_wrapper.original_env, 'env') else env_wrapper.original_env
                    if hasattr(underlying, 'highest_y'):
                        current_best = float(getattr(underlying, 'highest_y'))
                except Exception:
                    current_best = None

                if current_best is not None:
                    if current_best > max_height:
                        max_height = current_best
                    if (target_height is not None) and (current_best >= target_height):
                        steps_to_height_events.append(step_count)

                step_count += 1

            row: Dict[str, Any] = {
                "environment": env_name,
                "model": model_name,
                "num_agents": int(num_agents),
                "episode": int(ep + 1),
                "reward": episode_reward,
                "length": int(step_count),
                "max_height": float(max_height),
            }
            row["steps_to_height_events"] = [int(s) for s in steps_to_height_events] if steps_to_height_events else []
            all_episode_rows.append(row)

            if enable_tqdm and hasattr(iterator, 'set_postfix'):
                try:
                    iterator.set_postfix({'reward': f"{episode_reward:.2f}", 'max_h': f"{max_height:.2f}"})
                except Exception:
                    pass

        env_wrapper.close()

    return all_episode_rows


def _model_worker(args: Tuple):
    return _run_evaluations_for_model(*args)


def evaluate(
    config: Dict[str, Any],
    results_dir: str,
    models: Dict[str, str],
    metrics: List[str],
    num_agents_list: List[int],
) -> None:
    """
    BoxJump evaluator: loops over models and num_agents, runs episodes and records metrics.
    Supports optional GPU-parallel execution controlled by evaluation.parrallel.activate.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    env_name = config['environment']['name'] if 'environment' in config and 'name' in config['environment'] else config.get('environment_name', 'boxjump')
    base_env_kwargs = dict(config.get('environment', {}).get('env_kwargs', {}) or {})

    eval_cfg = config.get('evaluation', {})
    num_episodes = int(eval_cfg.get('episodes', 5))
    max_steps = int(eval_cfg.get('max_steps_per_episode', 1000))
    height_formula: Optional[str] = eval_cfg.get('height_formula')
    parallel_cfg = eval_cfg.get('parrallel', {}) or {}
    parallel_on = bool(parallel_cfg.get('activate', False))

    # Default to current env num_boxes if list not provided
    if not num_agents_list:
        default_agents = base_env_kwargs.get('num_boxes')
        num_agents_list = [int(default_agents)] if default_agents is not None else [8]

    # Storage for aggregate results across settings
    all_episode_rows: List[Dict[str, Any]] = []

    if parallel_on and len(models) > 1:
        gpu_count = (torch.cuda.device_count() if (torch is not None and torch.cuda.is_available()) else 0)
        model_items = list(models.items())
        worker_args: List[Tuple] = []
        for idx, (model_name, model_path) in enumerate(model_items):
            device_id = (idx % gpu_count) if gpu_count > 0 else None
            worker_args.append((
                config, results_dir, model_name, model_path, num_agents_list, metrics,
                base_env_kwargs, env_name, num_episodes, max_steps, height_formula, device_id, False
            ))

        ctx = mp.get_context('spawn')
        processes = min(len(worker_args), max(1, gpu_count) if gpu_count > 0 else len(worker_args))
        with ctx.Pool(processes=processes) as pool:
            for rows in pool.imap_unordered(_model_worker, worker_args):
                if rows:
                    all_episode_rows.extend(rows)
    else:
        for model_name, model_path in models.items():
            rows = _run_evaluations_for_model(
                config=config,
                results_dir=results_dir,
                model_name=model_name,
                model_path=model_path,
                num_agents_list=num_agents_list,
                metrics=metrics,
                base_env_kwargs=base_env_kwargs,
                env_name=env_name,
                num_episodes=num_episodes,
                max_steps=max_steps,
                height_formula=height_formula,
                device_id=None,
                enable_tqdm=True,
            )
            all_episode_rows.extend(rows)

    # Save raw rows
    with open(results_path / "episodes.json", "w") as f:
        json.dump(all_episode_rows, f, indent=2)

    # Build summaries per (model, num_agents)
    summary: List[Dict[str, Any]] = []
    from collections import defaultdict

    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_episode_rows:
        grouped[(r["model"], r["num_agents"])].append(r)

    for (model, num_agents), rows in grouped.items():
        rewards = np.array([r["reward"] for r in rows], dtype=float)
        lengths = np.array([r["length"] for r in rows], dtype=float)
        max_heights = np.array([r["max_height"] for r in rows], dtype=float)

        # Flatten steps_to_height events across episodes, exclude empties
        stm_events: List[int] = []
        for r in rows:
            stm_events.extend(r.get("steps_to_height_events", []) or [])
        stm = np.array(stm_events, dtype=float) if stm_events else np.array([], dtype=float)

        summary.append({
            "environment": env_name,
            "model": model,
            "num_agents": int(num_agents),
            "episodes": len(rows),
            "reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
            "reward_std": float(np.std(rewards)) if rewards.size else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths.size else 0.0,
            "length_std": float(np.std(lengths)) if lengths.size else 0.0,
            "max_height_mean": float(np.mean(max_heights)) if max_heights.size else 0.0,
            "max_height_std": float(np.std(max_heights)) if max_heights.size else 0.0,
            "steps_to_height_events_count": int(len(stm_events)),
            "steps_to_height_events_mean": float(np.mean(stm)) if stm.size else None,
            "steps_to_height_events_std": float(np.std(stm)) if stm.size else None,
        })

    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate box plots for requested metrics
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Prepare DataFrame-like structures without requiring pandas
        # For box plots, we need per-episode values grouped by (model, num_agents)
        def plot_metric(metric_key: str, title: str, filename: str, value_extractor):
            plot_rows = []
            for r in all_episode_rows:
                if metric_key == "steps_to_height":
                    events = r.get("steps_to_height_events", [])
                    for v in events:
                        plot_rows.append({
                            "model": r["model"],
                            "num_agents": r["num_agents"],
                            "value": float(v),
                        })
                else:
                    v = value_extractor(r)
                    plot_rows.append({
                        "model": r["model"],
                        "num_agents": r["num_agents"],
                        "value": float(v),
                    })

            if not plot_rows:
                return

            # Build plot arrays
            x_vals = sorted(set([pr["num_agents"] for pr in plot_rows]))
            models = sorted(set([pr["model"] for pr in plot_rows]))

            # Plot each model as separate series of boxplots along x=num_agents
            plt.figure(figsize=(10, 6))
            colors = sns.color_palette(n_colors=len(models))

            for idx, model in enumerate(models):
                data = []
                positions = []
                for xi, na in enumerate(x_vals):
                    vals = [pr["value"] for pr in plot_rows if pr["model"] == model and pr["num_agents"] == na]
                    if vals:
                        data.append(vals)
                        positions.append(xi + (idx+1) * (0.8/ (len(models)+1)))
                if data:
                    b = plt.boxplot(data, positions=positions, widths=0.15, patch_artist=True)
                    for patch in b['boxes']:
                        patch.set_facecolor(colors[idx])
                    plt.plot([], [], color=colors[idx], label=model)

            plt.xticks(range(len(x_vals)), [str(x) for x in x_vals])
            plt.xlabel("num agents")
            plt.ylabel(title)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_path / filename)
            plt.close()

        metrics_set = set(metrics or [])
        if 'reward' in metrics_set:
            plot_metric('reward', 'Reward', 'box_reward.png', lambda r: r['reward'])
        if 'max_height' in metrics_set:
            plot_metric('max_height', 'Max Height', 'box_max_height.png', lambda r: r['max_height'])
        if 'steps_to_height' in metrics_set:
            title = 'Steps to Target Height (events)'
            plot_metric('steps_to_height', title, 'box_steps_to_height.png', lambda r: 0.0)
    except Exception as e:
        # Plotting is optional; do not fail evaluation on plotting issues
        with open(results_path / "plot_errors.log", "a") as f:
            f.write(str(e) + "\n")


