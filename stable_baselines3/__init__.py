import importlib
import os
from typing import Any, Tuple


def _optional_import(module_path: str, attr_name: str) -> Tuple[Any, Exception]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name), None
    except Exception as exc:  # pragma: no cover - best effort for optional deps
        return None, exc


A2C, _A2C_IMPORT_ERROR = _optional_import("stable_baselines3.a2c", "A2C")
DDPG, _DDPG_IMPORT_ERROR = _optional_import("stable_baselines3.ddpg", "DDPG")
DQN, _DQN_IMPORT_ERROR = _optional_import("stable_baselines3.dqn", "DQN")
PPO, _PPO_IMPORT_ERROR = _optional_import("stable_baselines3.ppo", "PPO")
SAC, _SAC_IMPORT_ERROR = _optional_import("stable_baselines3.sac", "SAC")
TD3, _TD3_IMPORT_ERROR = _optional_import("stable_baselines3.td3", "TD3")
HerReplayBuffer, _HER_IMPORT_ERROR = _optional_import("stable_baselines3.her.her_replay_buffer", "HerReplayBuffer")

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    if HerReplayBuffer is None:
        raise ImportError(
            "Could not import `HerReplayBuffer` from stable_baselines3. "
            f"Original error: {_HER_IMPORT_ERROR}"
        )
    raise ImportError("Since Stable Baselines 2.1.0, `HER` is now `HerReplayBuffer`.")


def get_system_info(*args, **kwargs):
    from stable_baselines3.common.utils import get_system_info as _get_system_info

    return _get_system_info(*args, **kwargs)
