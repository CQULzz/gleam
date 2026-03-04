import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from gym.spaces import Box, Dict as DictSpace, MultiDiscrete


@dataclass
class Stage1LabEnvConfig:
    num_envs: int = 32
    device: str = "cuda:0"
    headless: bool = True
    backend: str = "isaaclab"
    buffer_size: int = 30
    grid_size: int = 128
    pose_size: int = 6
    max_episode_length: int = 512
    coverage_goal: float = 0.65
    step_scale_xy: float = 4.0
    coverage_reward_scale: float = 10.0
    collision_penalty: float = 0.02
    motion_height: float = 0.0


class _IsaacLabAppContext:
    """Best-effort Isaac Lab app launcher wrapper."""

    def __init__(self, headless: bool):
        self.launcher = None
        self.app = None
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise RuntimeError(
                "Isaac Lab backend was requested but `isaaclab` is not importable. "
                "Install the new stack first (see gleam_lab/scripts/setup_gleam_lab_env.sh)."
            ) from exc

        try:
            self.launcher = AppLauncher(headless=headless)
        except TypeError:
            self.launcher = AppLauncher({"headless": headless})
        self.app = getattr(self.launcher, "app", None)

    def close(self):
        if self.app is not None and hasattr(self.app, "close"):
            self.app.close()


class GLEAMStage1MockEnv:
    """Tensor-vectorized Stage-1 env used as Isaac Lab migration scaffold.

    This environment keeps the old GLEAM tensor API so SB3 rollout code remains unchanged.
    It uses a lightweight grid exploration dynamic to unblock training pipeline migration.
    """

    _is_tensor_env = True

    def __init__(self, cfg: Stage1LabEnvConfig):
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.grid_size = int(cfg.grid_size)
        self.buffer_size = int(cfg.buffer_size)
        self.pose_size = int(cfg.pose_size)
        self.max_episode_length = int(cfg.max_episode_length)
        self.max_episode_length_s = float(cfg.max_episode_length)
        self.device = torch.device(cfg.device)

        self._isaac_ctx = None
        if cfg.backend == "isaaclab":
            self._isaac_ctx = _IsaacLabAppContext(headless=cfg.headless)

        action_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
        action_high = np.array([128, 128, 0, 0, 0, 0], dtype=np.int64)
        self.action_space = MultiDiscrete(nvec=(action_high - action_low + 1).astype(np.int64))

        clip_pose_world_up = [float(self.grid_size - 1), float(self.grid_size - 1), cfg.motion_height, 0.0, 0.0, 0.0]
        clip_pose_world_low = [0.0, 0.0, cfg.motion_height, 0.0, 0.0, 0.0]
        pose_up_bound = np.tile(np.array(clip_pose_world_up, dtype=np.float32), self.buffer_size)
        pose_low_bound = np.tile(np.array(clip_pose_world_low, dtype=np.float32), self.buffer_size)

        self.observation_space = DictSpace(
            {
                "state": Box(
                    low=pose_low_bound,
                    high=pose_up_bound,
                    shape=(self.buffer_size * self.pose_size,),
                    dtype=np.float32,
                ),
                "ego_map_2D": Box(low=-1.0, high=2.0, shape=(self.grid_size * self.grid_size,), dtype=np.float32),
            }
        )

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras: Dict[str, torch.Tensor] = {
            "episode": {
                "episode_reward": torch.tensor(0.0, device=self.device),
                "rew_surface_coverage": torch.tensor(0.0, device=self.device),
                "rew_collision": torch.tensor(0.0, device=self.device),
            },
            "time_outs": self.time_out_buf,
        }

        self._positions_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._pose_buf = torch.zeros(
            (self.num_envs, self.buffer_size, self.pose_size), dtype=torch.float32, device=self.device
        )
        self._ego_map = torch.full(
            (self.num_envs, self.grid_size, self.grid_size), fill_value=-1.0, dtype=torch.float32, device=self.device
        )
        self._coverage_ratio = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(self._all_env_ids)

    def _update_pose_history(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self._pose_buf[env_ids, 1:, :] = self._pose_buf[env_ids, :-1, :].clone()
        self._pose_buf[env_ids, 0, :] = 0.0
        self._pose_buf[env_ids, 0, 0:2] = self._positions_xy[env_ids]
        self._pose_buf[env_ids, 0, 2] = self.cfg.motion_height
        self._pose_buf[env_ids, 0, 5] = self._yaw[env_ids]

    def _clip_positions(self, proposed_xy: torch.Tensor):
        low = 0.0
        high = float(self.grid_size - 1)
        collision = ((proposed_xy < low) | (proposed_xy > high)).any(dim=1)
        clipped = proposed_xy.clamp(min=low, max=high)
        return clipped, collision

    def _mark_visible(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        radius = 2
        xy_idx = self._positions_xy[env_ids].round().long()
        for local_i, env_id in enumerate(env_ids.tolist()):
            x = int(xy_idx[local_i, 0].item())
            y = int(xy_idx[local_i, 1].item())
            x0 = max(x - radius, 0)
            x1 = min(x + radius + 1, self.grid_size)
            y0 = max(y - radius, 0)
            y1 = min(y + radius + 1, self.grid_size)
            self._ego_map[env_id, x0:x1, y0:y1] = 1.0

    def _compute_coverage_ratio(self):
        explored = (self._ego_map == 1.0).float()
        return explored.mean(dim=(1, 2))

    def _build_observation(self):
        return {
            "state": self._pose_buf.reshape(self.num_envs, -1),
            "ego_map_2D": self._ego_map.reshape(self.num_envs, -1),
        }

    def _reset_envs(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        base_xy = torch.rand((env_ids.numel(), 2), device=self.device) * (self.grid_size * 0.5) + (self.grid_size * 0.25)
        self._positions_xy[env_ids] = base_xy
        self._yaw[env_ids] = 0.0
        self._pose_buf[env_ids] = 0.0
        self._ego_map[env_ids] = -1.0
        self._episode_reward_sum[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        self._mark_visible(env_ids)
        self._coverage_ratio[env_ids] = self._compute_coverage_ratio()[env_ids]
        self._update_pose_history(env_ids)

    def reset(self):
        self._reset_envs(self._all_env_ids)
        return self._build_observation()

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            action_tensor = torch.from_numpy(actions).to(self.device)
        elif torch.is_tensor(actions):
            action_tensor = actions.to(self.device)
        else:
            action_tensor = torch.as_tensor(actions, device=self.device)

        action_tensor = action_tensor.long()
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.view(1, -1)
        if action_tensor.shape[0] == 1 and self.num_envs > 1:
            action_tensor = action_tensor.repeat(self.num_envs, 1)
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got shape={tuple(action_tensor.shape)}")

        old_coverage = self._coverage_ratio.clone()

        # Use first two discrete bins to produce XY movement in [-step_scale_xy, step_scale_xy].
        centered = action_tensor[:, 0:2].float() - 64.0
        delta_xy = centered / 64.0 * float(self.cfg.step_scale_xy)
        proposed_xy = self._positions_xy + delta_xy
        self._positions_xy, collision = self._clip_positions(proposed_xy)

        self._mark_visible(self._all_env_ids)
        self._coverage_ratio = self._compute_coverage_ratio()
        self._update_pose_history(self._all_env_ids)

        rew_surface_coverage = (self._coverage_ratio - old_coverage) * float(self.cfg.coverage_reward_scale)
        rew_collision = collision.float() * float(self.cfg.collision_penalty)
        rewards = rew_surface_coverage - rew_collision

        self.episode_length_buf += 1
        self._episode_reward_sum += rewards

        time_outs = self.episode_length_buf >= self.max_episode_length
        coverage_goal_reached = self._coverage_ratio >= float(self.cfg.coverage_goal)
        dones = time_outs | coverage_goal_reached

        done_env_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            episode_reward = self._episode_reward_sum[done_env_ids].mean()
            episode_coverage = self._coverage_ratio[done_env_ids].mean()
        else:
            episode_reward = rewards.mean()
            episode_coverage = self._coverage_ratio.mean()

        self.time_out_buf = time_outs.clone()
        self.extras = {
            "episode": {
                "episode_reward": episode_reward,
                "rew_surface_coverage": rew_surface_coverage.mean(),
                "rew_collision": -rew_collision.mean(),
                "coverage_ratio": episode_coverage,
            },
            "time_outs": self.time_out_buf,
        }

        if done_env_ids.numel() > 0:
            self._reset_envs(done_env_ids)

        return self._build_observation(), rewards, dones, self.extras

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return [None] * self.num_envs
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed + i for i in range(self.num_envs)]

    def close(self):
        if self._isaac_ctx is not None:
            self._isaac_ctx.close()


def build_stage1_lab_config(args: argparse.Namespace) -> Stage1LabEnvConfig:
    return Stage1LabEnvConfig(
        num_envs=int(args.num_envs),
        device=str(args.device),
        headless=bool(args.headless),
        backend=str(args.backend),
        buffer_size=int(args.buffer_size),
        grid_size=int(args.grid_size),
        max_episode_length=int(args.n_steps),
    )
