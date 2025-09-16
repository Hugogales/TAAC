from typing import Callable


def get_evaluator(environment_name: str) -> Callable:
    """
    Return the environment-specific evaluator callable.

    Each evaluator must expose a function with signature:
        evaluate(
            config: dict,
            results_dir: str,
            models: dict[str, str],
            metrics: list[str],
            num_agents_list: list[int],
        ) -> None

    Raises ValueError if no evaluator exists for the environment.
    """
    name = (environment_name or "").lower()

    if name == "boxjump":
        from .boxjump import evaluate  # local import to avoid importing unnecessary deps
        return evaluate

    raise ValueError(f"No evaluator implemented for environment: {environment_name}")


