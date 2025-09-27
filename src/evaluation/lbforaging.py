from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

from src.env_wrapper import TAACEnvironmentWrapper
from src.model_factory import resolve_model_name, get_model_class


def _prepare_agent(config: Dict[str, Any], env_config: Dict[str, Any], model_name_override: Optional[str] = None):
    training_config = dict(config.get('training', {}) or {})
    if 'model' in config and isinstance(config['model'], dict):
        training_config.update(config['model'])
    model_name = model_name_override if model_name_override else resolve_model_name(config)
    ModelClass = get_model_class(model_name)
    agent = ModelClass(env_config, training_config, mode="test")
    return agent, model_name


def evaluate(
    config: Dict[str, Any],
    results_dir: str,
    models: Dict[str, str],
    metrics: List[str],
    num_agents_list: List[int],
) -> None:
    """
    Level-Based Foraging evaluator.
    Tracks: reward, episode length to termination, success rate, foods collected fraction.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    env_name = 'lbforaging'
    base_env_kwargs = dict(config.get('environment', {}).get('env_kwargs', {}) or {})

    eval_cfg = config.get('evaluation', {})
    num_episodes = int(eval_cfg.get('episodes', 5))
    max_steps = int(eval_cfg.get('max_steps_per_episode', 1000))

    # Default to provided players if list not provided
    if not num_agents_list:
        default_agents = base_env_kwargs.get('players') or base_env_kwargs.get('num_agents')
        num_agents_list = [int(default_agents)] if default_agents is not None else [2]

    all_episode_rows: List[Dict[str, Any]] = []

    for model_name, model_path in models.items():
        for num_agents in num_agents_list:
            env_kwargs = dict(base_env_kwargs)
            env_kwargs['players'] = int(num_agents)

            env_wrapper = TAACEnvironmentWrapper(
                env_name,
                apply_wrappers=config.get('environment', {}).get('apply_wrappers', True),
                **env_kwargs
            )

            env_config = env_wrapper.env_info
            agent, _resolved = _prepare_agent(config, env_config, model_name_override=model_name)

            # Device assignment
            try:
                if (torch is not None) and torch.cuda.is_available():
                    agent.assign_device(torch.device("cuda:0"))
                elif torch is not None:
                    agent.assign_device(torch.device("cpu"))
            except Exception:
                pass

            # Load model in test mode
            try:
                ok = agent.load_model(model_path, test=True)
            except Exception:
                ok = False
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

            iterator = tqdm(range(num_episodes), desc=f"{model_name} | players={num_agents}", leave=False)
            for ep in iterator:
                states, _ = env_wrapper.reset()
                episode_reward = 0.0
                step_count = 0
                done = False
                # Metrics bookkeeping
                spawned_count = None
                spawned_sum = None
                remaining_count = None
                remaining_sum = None
                success = False

                while not done and step_count < max_steps:
                    actions, _, _ = agent.get_actions(states)
                    states, rewards, done, info = env_wrapper.step(actions)
                    episode_reward += float(np.sum(rewards))
                    step_count += 1
                    # Read per-step info (same for all agents)
                    try:
                        first_info = next(iter(info.values())) if isinstance(info, dict) and info else {}
                    except Exception:
                        first_info = {}
                    spawned_count = first_info.get('foods_spawned_count', spawned_count)
                    spawned_sum = first_info.get('foods_spawned_sum', spawned_sum)
                    remaining_count = first_info.get('foods_remaining_count', remaining_count)
                    remaining_sum = first_info.get('foods_remaining_sum', remaining_sum)
                    if first_info.get('episode_success', False):
                        success = True

                all_episode_rows.append({
                    "environment": env_name,
                    "model": model_name,
                    "num_agents": int(num_agents),
                    "episode": int(ep + 1),
                    "reward": episode_reward,
                    "length": int(step_count),
                    "foods_spawned_count": int(spawned_count) if spawned_count is not None else None,
                    "foods_spawned_sum": float(spawned_sum) if spawned_sum is not None else None,
                    "foods_remaining_count": int(remaining_count) if remaining_count is not None else None,
                    "foods_remaining_sum": float(remaining_sum) if remaining_sum is not None else None,
                    "foods_collected_fraction": (
                        float(0.0) if (spawned_count in (None, 0) or remaining_count is None)
                        else float((spawned_count - remaining_count) / max(1, spawned_count))
                    ),
                    "success": bool(success),
                })

            env_wrapper.close()

    # Save raw rows
    with open(results_path / "episodes.json", "w") as f:
        json.dump(all_episode_rows, f, indent=2)

    # Build summaries per (model, num_agents)
    from collections import defaultdict
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_episode_rows:
        grouped[(r["model"], r["num_agents"])].append(r)

    summary: List[Dict[str, Any]] = []
    for (model, num_agents), rows in grouped.items():
        rewards = np.array([r["reward"] for r in rows], dtype=float)
        lengths = np.array([r["length"] for r in rows], dtype=float)
        successes = np.array([1.0 if r.get("success", False) else 0.0 for r in rows], dtype=float)
        frac_collected = np.array([
            r.get("foods_collected_fraction", 0.0) if r.get("foods_collected_fraction") is not None else 0.0
            for r in rows
        ], dtype=float)
        summary.append({
            "environment": env_name,
            "model": model,
            "num_agents": int(num_agents),
            "episodes": len(rows),
            "reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
            "reward_std": float(np.std(rewards)) if rewards.size else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths.size else 0.0,
            "length_std": float(np.std(lengths)) if lengths.size else 0.0,
            "success_rate": float(np.mean(successes)) if successes.size else 0.0,
            "foods_collected_fraction_mean": float(np.mean(frac_collected)) if frac_collected.size else 0.0,
            "foods_collected_fraction_std": float(np.std(frac_collected)) if frac_collected.size else 0.0,
        })

    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Optional plots (reward, length)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        def plot_metric(metric_key: str, title: str, filename: str):
            plot_rows = [{
                "model": r["model"],
                "num_agents": r["num_agents"],
                "value": float(r[metric_key])
            } for r in all_episode_rows]
            if not plot_rows:
                return
            x_vals = sorted(set([pr["num_agents"] for pr in plot_rows]))
            models = sorted(set([pr["model"] for pr in plot_rows]))

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

        if 'reward' in (metrics or []):
            plot_metric('reward', 'Reward', 'lbf_reward.png')
        # length is always available
        plot_metric('length', 'Episode Length', 'lbf_length.png')
        # success rate and foods collected fraction plots
        if 'success' in (metrics or ['success']):
            plot_metric('success', 'Success (1=yes)', 'lbf_success.png')
        if 'foods_collected_fraction' in (metrics or ['foods_collected_fraction']):
            plot_metric('foods_collected_fraction', 'Foods Collected Fraction', 'lbf_foods_collected_fraction.png')
    except Exception as e:
        with open(results_path / "plot_errors.log", "a") as f:
            f.write(str(e) + "\n")



