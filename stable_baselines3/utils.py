import datetime
import importlib
import os
from typing import Tuple

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_api_key_file(wandb_key_file=None):
    wandb_key_file = wandb_key_file or "wandb_api_key_file.txt"
    path = os.path.join(root, "wandb_utils", wandb_key_file)
    print("We are using this wandb key file: ", path)
    return path


def get_time_str():
    return datetime.datetime.now().strftime("%m%d_%H%M_%S")


def _resolve_legacy_isaac_types() -> Tuple[type, ...]:
    """Resolve legacy Isaac Gym env types lazily.

    This module is imported by many training paths (including Isaac Lab paths).
    Importing legacy Isaac Gym symbols eagerly can fail on newer Python/CUDA stacks.
    """
    resolved_types = []

    try:
        base_task_mod = importlib.import_module("legged_gym.env.base.base_task")
        resolved_types.append(getattr(base_task_mod, "BaseTask"))
    except Exception:
        pass

    try:
        wrapper_mod = importlib.import_module("gleam.wrapper.env_wrapper_gleam")
        resolved_types.append(getattr(wrapper_mod, "EnvWrapperGLEAM"))
    except Exception:
        pass

    return tuple(resolved_types)


def is_isaac_gym_env(env):
    # New tensor-based envs can mark themselves explicitly.
    if getattr(env, "_is_tensor_env", False):
        return True

    legacy_types = _resolve_legacy_isaac_types()
    if legacy_types and isinstance(env, legacy_types):
        return True

    # Soft fallback for wrappers/adapters that are intentionally isaac-like.
    class_name = env.__class__.__name__
    if class_name in {"EnvWrapperGLEAMLab", "GLEAMStage1LabEnv", "GLEAMStage1MockEnv"}:
        return True

    return False
