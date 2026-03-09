from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

from gleam_lab.app import repo_root


@dataclass
class RewardScalesCfg:
    surface_coverage_2d: float = 1000.0
    termination: float = 50.0
    collision: float = -100.0


@dataclass
class RewardsCfg:
    scales: RewardScalesCfg = field(default_factory=RewardScalesCfg)
    only_positive_rewards: bool = False
    max_contact_force: float = 100.0


@dataclass
class NormalizationCfg:
    init_action: tuple[int, int, int, int, int, int] = (64, 64, 0, 0, 0, 0)
    clip_actions_up: tuple[int, int, int, int, int, int] = (128, 128, 0, 0, 0, 0)
    clip_actions_low: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    clip_observations: float = 255.0


@dataclass
class VisualInputCfg:
    camera_width: int = 256
    camera_height: int = 32
    horizontal_fov: float = 90.0
    stack: int = 30
    supersampling_horizontal: int = 1
    supersampling_vertical: int = 1
    normalization: bool = True
    far_plane: float = 2_000_000.0
    near_plane: float = 0.0010000000474974513


@dataclass
class EnvCfg:
    num_observations: int = 6
    episode_length_s: float = 20.0
    num_actions: int = 6
    env_spacing: float = 20.0
    send_timeouts: bool = True


@dataclass
class TerrainCfg:
    curriculum: bool = False
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class CommandsCfg:
    curriculum: bool = False


@dataclass
class TerminationCfg:
    collision: bool = True
    max_step_done: bool = True


@dataclass
class ConfigGLEAMLabStage1:
    position_use_polar_coordinates: bool = False
    direction_use_vector: bool = False
    debug_save_image_tensor: bool = False
    debug_save_path: Optional[str] = None
    max_episode_length: int = 500
    num_sampled_point: int = 5000
    return_visual_observation: bool = True
    debug_viz: bool = False
    headless: bool = True
    device: str = "cuda:0"
    eval_device: str = "cuda:0"
    seed: int = 1
    data_root: str = f"{repo_root()}/data_gleam"
    dataset_name: str = "stage1_512"
    num_scene_override: Optional[int] = None
    sim_dt: float = 0.02
    motion_height: float = 1.5
    ego_cell_size: float = 0.1
    recent_num: int = 10
    ratio_threshold_term: float = 0.98
    ratio_threshold_rew: float = 0.75
    visualize_flag: bool = False
    env: EnvCfg = field(default_factory=EnvCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    visual_input: VisualInputCfg = field(default_factory=VisualInputCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)


def build_stage1_lab_config(args: argparse.Namespace) -> ConfigGLEAMLabStage1:
    cfg = ConfigGLEAMLabStage1()
    cfg.headless = bool(args.headless)
    cfg.device = str(args.sim_device)
    cfg.eval_device = str(args.eval_device)
    cfg.seed = int(args.seed)
    cfg.env.num_envs = int(args.num_envs)
    cfg.visual_input.stack = int(args.buffer_size)
    cfg.rewards.scales.surface_coverage_2d = float(args.surface_coverage) * 1000.0
    cfg.rewards.only_positive_rewards = bool(args.only_positive_rewards)
    cfg.rewards.max_contact_force = float(args.max_contact_force)
    cfg.data_root = str(args.data_root)
    cfg.dataset_name = str(args.dataset_name)
    cfg.num_scene_override = args.num_scene_override
    return cfg
