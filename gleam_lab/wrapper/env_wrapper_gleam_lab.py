"""GLEAM Lab env wrapper that flattens dict observations into grid-observation tensors."""

import gym
import numpy as np
import torch
from gym import spaces


def flatten_observations(observation_dict, key_sequence):
    observations = []
    for key in key_sequence:
        observations.append(observation_dict[key])
    return torch.concat(observations, dim=-1)


def flatten_observation_spaces(observation_spaces, key_sequence):
    assert isinstance(key_sequence, list)
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
    _is_tensor_env = True

    def __init__(self, gym_env):
        self._gym_env = gym_env
        self.observation_space = flatten_observation_spaces(
            observation_spaces=self._gym_env.observation_space,
            key_sequence=["state", "ego_map_2D"],
        )
        self.action_space = self._gym_env.action_space
        self.num_envs = self._gym_env.num_envs

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def reset(self):
        observation = self._gym_env.reset()
        return flatten_observations(observation_dict=observation, key_sequence=["state", "ego_map_2D"])

    def step(self, action):
        observation_dict, reward, done, info = self._gym_env.step(action)
        flat_obs = flatten_observations(observation_dict=observation_dict, key_sequence=["state", "ego_map_2D"])
        return flat_obs, reward, done, info

    def close(self):
        self._gym_env.close()
