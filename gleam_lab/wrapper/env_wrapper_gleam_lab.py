from __future__ import annotations

import gym
import numpy as np
import torch
from gym import spaces


def flatten_observations(observation_dict, key_sequence):
    observations = [observation_dict[key] for key in key_sequence]
    return torch.concat(observations, dim=-1)


def flatten_observation_spaces(observation_spaces, key_sequence):
    lower_bound = []
    upper_bound = []
    for key in key_sequence:
        value = observation_spaces.spaces[key]
        if isinstance(value, spaces.Box):
            lower_bound.append(np.asarray(value.low).flatten())
            upper_bound.append(np.asarray(value.high).flatten())
    lower_bound = np.concatenate(lower_bound)
    upper_bound = np.concatenate(upper_bound)
    return spaces.Box(np.array(lower_bound, dtype=np.float32), np.array(upper_bound, dtype=np.float32), dtype=np.float32)


class EnvWrapperGLEAMLab(gym.Env):
    """Flatten the dict observation to the original Stage-1 SB3 input layout."""

    def __init__(self, gym_env, observation_excluded=()):
        self.observation_excluded = observation_excluded
        self._gym_env = gym_env
        self.observation_space = flatten_observation_spaces(self._gym_env.observation_space, ["state", "ego_map_2D"])
        self.action_space = self._gym_env.action_space

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def reset(self):
        observation = self._gym_env.reset()
        return flatten_observations(observation, ["state", "ego_map_2D"])

    def step(self, action):
        observation_dict, reward, done, info = self._gym_env.step(action)
        return flatten_observations(observation_dict, ["state", "ego_map_2D"]), reward, done, info

    def render(self, mode="human"):
        return self._gym_env.render(mode)

    def close(self):
        self._gym_env.close()
