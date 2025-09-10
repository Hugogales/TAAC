import importlib
from typing import Type, Tuple


def resolve_model_name(config) -> str:
    """
    Determine which model/algorithm to use based on the config.
    Priority:
      1) config['algorithm']['name']
      2) config['model']['name']
      3) default 'TAAC'
    """
    try:
        algo = (config.get('algorithm') or {}).get('name')
        if algo:
            return str(algo).strip().upper()
    except Exception:
        pass

    try:
        model_name = (config.get('model') or {}).get('name')
        if model_name:
            return str(model_name).strip().upper()
    except Exception:
        pass

    return 'TAAC'


def get_model_class(name: str) -> Type:
    """
    Map a string name to a model class in src.AI.*
    Supported: TAAC, PPO, MAAC
    """
    name_upper = str(name).strip().upper()

    if name_upper == 'TAAC':
        module = importlib.import_module('src.AI.TAAC')
        return getattr(module, 'TAAC')
    if name_upper == 'PPO':
        module = importlib.import_module('src.AI.PPO')
        return getattr(module, 'PPO')
    if name_upper == 'MAAC':
        module = importlib.import_module('src.AI.MAAC')
        return getattr(module, 'MAAC')

    raise ValueError(f"Unknown model/algorithm name: {name}")


