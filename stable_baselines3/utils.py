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

    This module is imported by both legacy Isaac Gym and Isaac Lab training paths.
    Eagerly importing Isaac Gym-only symbols breaks on modern Python/CUDA stacks.
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

    try:
        wrapper_mod = importlib.import_module("gleam_lab.wrapper.env_wrapper_gleam_lab")
        resolved_types.append(getattr(wrapper_mod, "EnvWrapperGLEAMLab"))
    except Exception:
        pass

    return tuple(resolved_types)


def is_isaac_gym_env(env):
    if getattr(env, "_is_tensor_env", False):
        return True

    legacy_types = _resolve_legacy_isaac_types()
    if legacy_types and isinstance(env, legacy_types):
        return True

    class_name = env.__class__.__name__
    if class_name in {"EnvWrapperGLEAMLab", "Env_GLEAM_Stage1_Lab", "GLEAMStage1LabEnv", "GLEAMStage1MockEnv"}:
        return True

    return False
