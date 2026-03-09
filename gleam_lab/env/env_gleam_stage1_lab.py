from __future__ import annotations

import math
import os
import random
from collections import deque

import gym
import numpy as np
import torch
import trimesh
from gym.spaces import Box, Dict, MultiDiscrete
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.views import XformPrimView
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from gleam_lab.app import enable_urdf_importer_for_isaacsim_51
from gleam_lab.env.config_gleam_lab import ConfigGLEAMLabStage1
from gleam_lab.env.grid_ops import (
    bresenham_2d_batched,
    build_env_origins,
    compute_frontier_map,
    create_e2w_from_poses,
    discretize_prob_map,
    extract_ego_maps,
    pose_coord_to_2d_idx,
    quat_wxyz_from_euler_xyz,
    rotmat_from_quat_wxyz,
    scanned_pts_to_2d_idx_batched,
    transform_from_pose,
)
from gleam_lab.env.path_ops import bfs_path_lengths


class Env_GLEAM_Stage1_Lab(gym.Env):
    """Stage-1 Isaac Lab environment that preserves the original GLEAM semantics."""

    metadata = {"render.modes": []}

    def __init__(self, cfg: ConfigGLEAMLabStage1):
        super().__init__()
        self.cfg = cfg
        self.headless = cfg.headless
        self.device = torch.device(cfg.device)
        self.num_envs = int(cfg.env.num_envs)
        self.max_episode_length = int(cfg.max_episode_length)
        self.max_episode_length_s = float(cfg.env.episode_length_s)
        self.buffer_size = int(cfg.visual_input.stack)
        self.motion_height = float(cfg.motion_height)
        self.ego_cell_size = float(cfg.ego_cell_size)
        self.visualize_flag = bool(cfg.visualize_flag)
        self.debug = os.environ.get("GLEAM_LAB_DEBUG", "0") == "1"
        self.grid_size = 128
        self.scene_quat = torch.tensor([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.main_camera_angles = [0.0, 90.0, 180.0, 270.0]
        self.num_cam = len(self.main_camera_angles)
        self.depth_sense_dist = -50.0
        self.num_scene = self._resolve_num_scene()
        assert self.num_scene >= self.num_envs, "num_scene should be larger than num_envs"
        assert self.num_scene % self.num_envs == 0, "num_scene must be divisible by num_envs"

        self.sim = None
        self.scene_view = None
        self.base_view = None
        self.main_cameras: list[Camera] = []
        self.collision_camera: Camera | None = None
        self.init_action_tensor = torch.tensor(self.cfg.normalization.init_action, dtype=torch.int64, device=self.device)

        self._set_random_seed(cfg.seed)
        self._create_simulation()
        self._init_buffers()
        self.update_observation_space()

    @property
    def sim_dt(self) -> float:
        return float(self.cfg.sim_dt)

    def _resolve_num_scene(self) -> int:
        default_scene_count = int(self.cfg.dataset_name.split("_")[-1])
        return int(self.cfg.num_scene_override) if self.cfg.num_scene_override else default_scene_count

    def _set_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def urdf_dir(self) -> str:
        return os.path.join(self.cfg.data_root, f"train_{self.cfg.dataset_name}", "urdf")

    @property
    def gt_dir(self) -> str:
        return os.path.join(self.cfg.data_root, f"train_{self.cfg.dataset_name}", "gt")

    def _create_simulation(self):
        self._debug("create_simulation:start")
        enable_urdf_importer_for_isaacsim_51()
        sim_utils.create_new_stage()
        self.sim = SimulationContext(SimulationCfg(dt=self.sim_dt, use_fabric=False))
        self.stage = self.sim.stage
        ground_cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(200.0, 200.0))
        ground_cfg.func("/World/ground", ground_cfg)
        sim_utils.create_prim(
            "/World/envs",
            prim_type="Xform",
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            stage=self.stage,
        )
        self.env_origins = build_env_origins(self.num_envs, self.cfg.env.env_spacing, self.device)
        self.scene_per_env = self.num_scene // self.num_envs
        self.inactive_xyz = torch.tensor(
            [
                self.env_origins[:, 0].max().item() + 15.0,
                self.env_origins[:, 1].max().item() + 15.0,
                self.env_origins[:, 2].max().item(),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        for env_idx in range(self.num_envs):
            env_path = self._env_path(env_idx)
            sim_utils.create_prim(
                env_path,
                prim_type="Xform",
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                stage=self.stage,
            )
            sim_utils.create_prim(
                self._base_path(env_idx),
                prim_type="Xform",
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                stage=self.stage,
            )
            self._debug(f"spawn_env:{env_idx}")
            self._spawn_scenes_for_env(env_idx)
            self._spawn_cameras_for_env(env_idx)

        self._debug("create_simulation:update_stage_after_spawn")
        sim_utils.update_stage()
        sim_utils.update_stage()
        self._debug("create_simulation:create_camera_interfaces")
        self._create_camera_interfaces()
        self._debug("create_simulation:sim_reset")
        self.sim.reset()
        self._debug("create_simulation:warm_up")
        self._warm_up_cameras(5)
        self._debug("create_simulation:create_scene_view")
        self.scene_view = XformPrimView("/World/envs/env_.*/scene_.*", device=self.device, stage=self.stage)
        self.base_view = XformPrimView("/World/envs/env_.*/base", device=self.device, stage=self.stage)
        self._debug("create_simulation:done")

    def _env_path(self, env_idx: int) -> str:
        return f"/World/envs/env_{env_idx:04d}"

    def _base_path(self, env_idx: int) -> str:
        return f"{self._env_path(env_idx)}/base"

    def _scene_path(self, scene_idx: int) -> str:
        env_idx = scene_idx // self.scene_per_env
        return f"{self._env_path(env_idx)}/scene_{scene_idx:04d}"

    def _spawn_scenes_for_env(self, env_idx: int):
        active_origin = self.env_origins[env_idx].cpu().tolist()
        inactive_origin = self.inactive_xyz.cpu().tolist()
        mesh_cfg = MeshCfg()
        for idx in range(self.scene_per_env):
            scene_idx = env_idx * self.scene_per_env + idx
            mesh = trimesh.load_mesh(
                os.path.join(self.cfg.data_root, f"train_{self.cfg.dataset_name}", "obj", f"scene_{scene_idx}.obj"),
                process=False,
            )
            translation = tuple(active_origin) if idx == 0 else tuple(inactive_origin)
            _spawn_mesh_geom_from_mesh(
                self._scene_path(scene_idx),
                mesh_cfg,
                mesh,
                translation=translation,
                orientation=tuple(self.scene_quat.cpu().tolist()),
                stage=self.stage,
            )

    def _spawn_cameras_for_env(self, env_idx: int):
        env_path = self._env_path(env_idx)
        base_path = self._base_path(env_idx)
        main_spawn = self._make_camera_spawn_cfg(self.cfg.visual_input.camera_width, self.cfg.visual_input.camera_height, self.cfg.visual_input.horizontal_fov)
        col_fov = math.atan(40 / self.cfg.visual_input.camera_width) / math.pi * 180.0
        col_spawn = self._make_camera_spawn_cfg(40, 40, col_fov)
        for cam_idx in range(self.num_cam):
            yaw = math.radians(self.main_camera_angles[cam_idx])
            quat_world = quat_wxyz_from_euler_xyz(
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([yaw], dtype=torch.float32),
            )[0]
            quat_gl = convert_camera_frame_orientation_convention(
                quat_world.unsqueeze(0), origin="world", target="opengl"
            )[0]
            main_spawn.func(
                f"{base_path}/camera_main_{cam_idx}",
                main_spawn,
                translation=(0.0, 0.0, 0.1),
                orientation=tuple(quat_gl.tolist()),
            )
        col_spawn.func(f"{env_path}/camera_col", col_spawn, translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

    def _make_camera_spawn_cfg(self, width: int, height: int, horizontal_fov_deg: float):
        fov_x = math.radians(horizontal_fov_deg)
        # Preserve square-pixel intrinsics: derive vertical FoV from horizontal FoV through aspect ratio.
        fov_y = 2.0 * math.atan((height / width) * math.tan(0.5 * fov_x))
        focal_x = 0.5 * width / math.tan(0.5 * fov_x)
        focal_y = 0.5 * height / math.tan(0.5 * fov_y)
        intrinsic = [focal_x, 0.0, width / 2.0, 0.0, focal_y, height / 2.0, 0.0, 0.0, 1.0]
        return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic,
            width=width,
            height=height,
            clipping_range=(self.cfg.visual_input.near_plane, self.cfg.visual_input.far_plane),
        )

    def _create_camera_interfaces(self):
        self.main_cameras = []
        for cam_idx in range(self.num_cam):
            cfg = CameraCfg(
                height=self.cfg.visual_input.camera_height,
                width=self.cfg.visual_input.camera_width,
                prim_path=f"/World/envs/env_.*/base/camera_main_{cam_idx}",
                update_period=0.0,
                data_types=["distance_to_image_plane"],
                spawn=None,
                update_latest_camera_pose=True,
            )
            self.main_cameras.append(Camera(cfg))
        self.collision_camera = Camera(
            CameraCfg(
                height=40,
                width=40,
                prim_path="/World/envs/env_.*/camera_col",
                update_period=0.0,
                data_types=["distance_to_image_plane"],
                spawn=None,
                update_latest_camera_pose=True,
            )
        )

    def _warm_up_cameras(self, steps: int):
        for step_idx in range(steps):
            self._debug(f"warm_up:sim_step:{step_idx}")
            self.sim.step()
        self._debug("warm_up:update_begin")
        for cam_idx, camera in enumerate(self.main_cameras):
            self._debug(f"warm_up:update_main:{cam_idx}")
            camera.update(self.sim_dt)
        self._debug("warm_up:update_collision")
        self.collision_camera.update(self.sim_dt)
        self._debug("warm_up:update_done")

    def _init_buffers(self):
        self._debug("init_buffers:start")
        self._debug("init_buffers:load_all:start")
        self._init_load_all()
        self._debug("init_buffers:load_all:done")
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reward_scales = {
            "surface_coverage_2d": self.cfg.rewards.scales.surface_coverage_2d * self.sim_dt,
            "collision": self.cfg.rewards.scales.collision * self.sim_dt,
            "termination": self.cfg.rewards.scales.termination * self.sim_dt,
        }
        self.reward_names = ["surface_coverage_2d", "collision"]
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in self.reward_scales.keys()
        }
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras = {}

        self.env_to_scene = torch.tensor([env_idx * self.scene_per_env for env_idx in range(self.num_envs)], dtype=torch.long, device=self.device)
        self.active_scene_ids = self.env_to_scene.tolist()
        self.inactive_scene_ids = [scene_idx for scene_idx in range(self.num_scene) if scene_idx not in self.active_scene_ids]

        self.range_gt_scenes = self.range_gt[self.env_to_scene].clone()
        self.voxel_size_gt_scenes = self.voxel_size_gt[self.env_to_scene].clone()
        self.num_valid_pixel_gt_scenes = self.num_valid_pixel_gt[self.env_to_scene].clone()
        self.layout_maps_height_scenes = self.layout_maps_height[self.env_to_scene].clone()

        self.clip_pose_world_low = torch.cat(
            [self.range_gt[:, 1::2], torch.zeros((self.num_scene, 3), dtype=torch.float32, device=self.device)], dim=1
        )
        self.clip_pose_world_up = torch.cat(
            [self.range_gt[:, ::2], torch.tensor([0.0, 0.0, 2 * torch.pi], dtype=torch.float32, device=self.device).repeat(self.num_scene, 1)], dim=1
        )

        self.actions = torch.tensor(self.cfg.normalization.init_action, dtype=torch.int64, device=self.device).repeat(self.num_envs, 1)
        self.action_size = self.actions.shape[1]
        self.clip_actions_low = torch.tensor(self.cfg.normalization.clip_actions_low, dtype=torch.int64, device=self.device)
        self.clip_actions_up = torch.tensor(self.cfg.normalization.clip_actions_up, dtype=torch.int64, device=self.device)
        self.ratio_threshold_term = float(self.cfg.ratio_threshold_term)
        self.ratio_threshold_rew = float(self.cfg.ratio_threshold_rew)
        self.recent_num = int(self.cfg.recent_num)
        self.collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reward_layout_ratio_buf = deque(maxlen=self.buffer_size)
        self.reward_layout_ratio_buf.extend(self.buffer_size * [torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)])

        self._debug("init_buffers:camera_intrinsics:start")
        self.inv_intri = torch.linalg.inv(self.get_camera_intrinsics()).to(self.device).to(torch.float32)
        self._debug("init_buffers:camera_intrinsics:done")
        xs = torch.linspace(0, self.cfg.visual_input.camera_width - 1, self.cfg.visual_input.camera_width, dtype=torch.float32, device=self.device)
        ys = torch.linspace(0, self.cfg.visual_input.camera_height - 1, self.cfg.visual_input.camera_height, dtype=torch.float32, device=self.device)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        norm_coord_pixel = torch.stack([xs, ys], dim=-1)
        norm_coord_pixel = torch.concat((norm_coord_pixel, torch.ones_like(norm_coord_pixel[..., :1], device=self.device)), dim=-1).view(-1, 3)
        self.norm_coord_pixel_around = norm_coord_pixel.repeat(self.num_cam, 1)
        self.blender2opencv = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        self.frontier_kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.env_idx_tensor = torch.arange(self.num_envs, device=self.device)

        self.poses = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.poses[:, 2] = self.motion_height
        self.poses_idx = torch.tensor([self.grid_size // 2, self.grid_size // 2], dtype=torch.int32, device=self.device).repeat(self.num_envs, 1)
        self.poses_idx_old = self.poses_idx.clone()
        self.pose_buf = [self.poses.clone() for _ in range(10)]
        self.world_pose_buf = torch.zeros((self.num_envs, self.buffer_size, 6), dtype=torch.float32, device=self.device)
        self.ego_pose_buf = torch.zeros((self.num_envs, self.buffer_size, 6), dtype=torch.float32, device=self.device)
        self.scanned_gt_map = torch.zeros((self.num_envs, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)
        self.prob_map = torch.zeros_like(self.scanned_gt_map)
        self.ego_prob_maps = torch.zeros_like(self.scanned_gt_map)
        self.occ_maps_tri_cls = torch.zeros_like(self.scanned_gt_map)
        self.depth_processed = torch.zeros((self.num_envs, self.num_cam, self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width), dtype=torch.float32, device=self.device)
        self.depth_processed_col = torch.zeros((self.num_envs, 40, 40), dtype=torch.float32, device=self.device)
        self.c2ws = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_cam, 1, 1)
        self._debug("init_buffers:reset_idx:start")
        self.reset_idx(torch.arange(self.num_envs, device=self.device), switch_scene=False)
        self._debug("init_buffers:reset_idx:done")
        self._debug("init_buffers:done")

    def _debug(self, message: str):
        if self.debug:
            print(f"[Stage1Lab] {message}", flush=True)

    def _init_load_all(self):
        voxel_path = os.path.join(self.gt_dir, f"{self.cfg.dataset_name}_{self.grid_size}_voxel_size_gt.pt")
        range_path = os.path.join(self.gt_dir, f"{self.cfg.dataset_name}_{self.grid_size}_range_gt.pt")
        occ_path = os.path.join(self.gt_dir, f"{self.cfg.dataset_name}_{self.grid_size}_occ_map_height_1d5_gt.pt")
        init_path = os.path.join(self.gt_dir, f"{self.cfg.dataset_name}_{self.grid_size}_init_map_1d5.pt")
        self.voxel_size_gt = torch.load(voxel_path, map_location=self.device)[: self.num_scene]
        self.range_gt = torch.load(range_path, map_location=self.device)[: self.num_scene]
        self.layout_maps_height = torch.load(occ_path, map_location=self.device)[: self.num_scene].to(torch.float16) / 255.0
        self.num_valid_pixel_gt = self.layout_maps_height.sum(dim=(1, 2))
        init_maps = torch.load(init_path, map_location=self.device)[: self.num_scene] / 255.0
        self.init_maps_list = [
            (torch.nonzero(init_maps[idx]) / (self.grid_size - 1) * 2 - 1) * self.range_gt[idx, :4:2]
            for idx in range(self.num_scene)
        ]

    def update_observation_space(self):
        action_space_size = (self.clip_actions_up - self.clip_actions_low + 1).cpu().numpy().astype(np.int64)
        self.action_space = MultiDiscrete(nvec=action_space_size)
        x_max = self.range_gt[:, 0].max().item()
        x_min = self.range_gt[:, 1].min().item()
        y_max = self.range_gt[:, 2].max().item()
        y_min = self.range_gt[:, 3].min().item()
        pose_up_bound = np.tile([x_max, y_max, self.motion_height, 0.0, 0.0, 0.0], self.buffer_size).astype(np.float32)
        pose_low_bound = np.tile([x_min, y_min, self.motion_height, 0.0, 0.0, 0.0], self.buffer_size).astype(np.float32)
        self.observation_space = Dict(
            {
                "state": Box(low=pose_low_bound, high=pose_up_bound, shape=(self.buffer_size * 6,), dtype=np.float32),
                "ego_map_2D": Box(low=-1.0, high=2.0, shape=(self.grid_size * self.grid_size,), dtype=np.float32),
            }
        )

    def get_camera_intrinsics(self) -> torch.Tensor:
        return self.main_cameras[0].data.intrinsic_matrices[0].clone().to(torch.float32)

    def _set_base_pose(self, poses: torch.Tensor):
        position_world = poses[:, :3] + self.env_origins
        roll = poses[:, 3]
        pitch = poses[:, 4]
        base_yaw = poses[:, 5]
        quat_world = quat_wxyz_from_euler_xyz(roll, pitch, base_yaw)
        self.base_view.set_world_poses(position_world, quat_world)

    def _step_sim(self):
        self.sim.step()
        for camera in self.main_cameras:
            camera.update(self.sim_dt)
        self.collision_camera.update(self.sim_dt)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.actions = torch.clip(self.actions, self.clip_actions_low, self.clip_actions_up)
        self.e2w = create_e2w_from_poses(self.poses, self.device)
        self.update_pose(self.actions, self.e2w)
        self._set_base_pose(self.poses)
        self._step_sim()
        obs = self.post_physics_step(if_reset=True)
        return obs

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        elif not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device)
        self.actions = torch.clip(actions.long(), self.clip_actions_low, self.clip_actions_up)
        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if env_ids:
            self.actions[env_ids] = self.init_action_tensor
            self.e2w[env_ids] = create_e2w_from_poses(self.poses[env_ids], self.device)
        current_poses = self.poses.clone()
        self.update_pose(self.actions, self.e2w)
        current_move_xyz = torch.cat([self.move_dist, torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)], dim=1)
        self.set_collision_cam_pose(current_poses, current_move_xyz)
        self._set_base_pose(self.poses)
        self._step_sim()
        return self.post_physics_step()

    def update_pose(self, ego_actions: torch.Tensor, e2w: torch.Tensor):
        cur_poses = self.poses.clone()
        ego_move_xyz = ego_actions[:, :3].clone().to(torch.float32)
        ego_move_xyz[:, :2] = (ego_move_xyz[:, :2] - self.grid_size / 2) * self.voxel_size_gt_scenes[:, :2]
        ego_move_xyz = ego_move_xyz.unsqueeze(-1)
        world_move_xyz = torch.bmm(e2w, ego_move_xyz).squeeze(-1)
        tar_poses = self.poses.clone()
        tar_poses[:, :3] += world_move_xyz
        tar_pose_idx = pose_coord_to_2d_idx(tar_poses[:, :2].clone(), self.range_gt_scenes, self.voxel_size_gt_scenes, self.grid_size).to(torch.long)
        self.tar_no_collision = self.scanned_gt_map[self.env_idx_tensor, tar_pose_idx[:, 0], tar_pose_idx[:, 1]] != 1.0
        self.poses[self.tar_no_collision, :3] += world_move_xyz[self.tar_no_collision]
        self.clip_pose_map_bound()
        self.move_dist = self.poses[:, :3] - cur_poses[:, :3]

    def clip_pose_map_bound(self):
        self.poses = torch.clip(self.poses, self.clip_pose_world_low[self.env_to_scene], self.clip_pose_world_up[self.env_to_scene])
        self.poses[:, -1] = (self.poses[:, -1] + 2 * torch.pi) % (2 * torch.pi)

    def set_collision_cam_pose(self, cur_poses: torch.Tensor, movements: torch.Tensor):
        cur_positions = cur_poses[:, :3] + self.env_origins
        tar_positions = cur_positions + movements[:, :3]
        self.collision_camera.set_world_poses_from_view(cur_positions, tar_positions)

    def post_process_main_camera_tensor(self):
        def process_depth(camera_output: torch.Tensor) -> torch.Tensor:
            depth = camera_output.squeeze(-1).to(torch.float32)
            depth = torch.clamp(torch.nan_to_num(depth, neginf=0.0).abs(), min=self.depth_sense_dist)
            return depth

        main_depth = []
        c2w_list = []
        for cam_idx in range(self.num_cam):
            cam_data = self.main_cameras[cam_idx].data
            main_depth.append(process_depth(cam_data.output["distance_to_image_plane"]))
            quat_gl = cam_data.quat_w_opengl
            pos_w = cam_data.pos_w
            c2w_gl = transform_from_pose(pos_w, quat_gl)
            c2w = torch.matmul(c2w_gl, self.blender2opencv.unsqueeze(0))
            c2w[:, :3, 3] -= self.env_origins
            c2w_list.append(c2w)
        self.depth_processed = torch.stack(main_depth, dim=1)
        self.depth_processed_col = process_depth(self.collision_camera.data.output["distance_to_image_plane"])
        self.c2ws = torch.stack(c2w_list, dim=1)

    def update_observation(self):
        self.update_pose_buf()
        self.update_occ_map_2d()

    def update_pose_buf(self):
        actual_poses = self.poses.clone()
        for env_idx in range(self.num_envs):
            num_step = int(self.cur_episode_length[env_idx].item())
            if num_step != 0 and num_step <= self.buffer_size - 1:
                self.world_pose_buf[env_idx, 1 : num_step + 1] = self.world_pose_buf[env_idx, :num_step].clone()
            else:
                self.world_pose_buf[env_idx, 1:] = self.world_pose_buf[env_idx, : self.buffer_size - 1].clone()
            self.world_pose_buf[env_idx, 0] = actual_poses[env_idx]
            self.ego_pose_buf[env_idx, : num_step + 1] = self.world_pose_buf[env_idx, : num_step + 1] - self.world_pose_buf[env_idx, 0].clone()
        self.pose_buf.pop(0)
        self.pose_buf.append(actual_poses.clone())

    def back_projection_stack(self) -> torch.Tensor:
        num_env = self.depth_processed.shape[0]
        num_cam = self.depth_processed.shape[1]
        h = self.cfg.visual_input.camera_height
        w = self.cfg.visual_input.camera_width
        depth_maps = self.depth_processed.reshape(num_env, -1)
        coords_pixel = torch.einsum("ij,jk->ijk", depth_maps, self.norm_coord_pixel_around)
        coords_cam = torch.einsum("ij,nkj->nki", self.inv_intri, coords_pixel)
        coords_cam_homo = torch.concat((coords_cam, torch.ones_like(coords_cam[..., :1], device=self.device)), dim=-1)
        coords_cam_homo = coords_cam_homo.view(num_env, num_cam, h * w, 4)
        coords_world_around = torch.matmul(self.c2ws.unsqueeze(2), coords_cam_homo.unsqueeze(-1)).squeeze(-1)
        return coords_world_around[..., :3].reshape(num_env, num_cam * h * w, 3)

    def update_occ_map_2d(self):
        pts_target = self.back_projection_stack()
        ray_env_ids, pts_idx_all = scanned_pts_to_2d_idx_batched(
            pts_target, self.range_gt_scenes, self.voxel_size_gt_scenes, self.motion_height, self.grid_size
        )
        self.poses_idx_old = self.poses_idx.to(torch.int32).clone()
        pose_idx = pose_coord_to_2d_idx(self.poses[:, :2].clone(), self.range_gt_scenes, self.voxel_size_gt_scenes, self.grid_size)
        self.poses_idx = pose_idx.to(torch.int32).clone()
        current_pose_idx = self.poses_idx.to(torch.long)
        self.current_pose_state = self.occ_maps_tri_cls[self.env_idx_tensor, current_pose_idx[:, 0], current_pose_idx[:, 1]].clone()
        self.e2w = create_e2w_from_poses(self.poses, self.device)
        if pts_idx_all.numel() > 0:
            ray_cast_env_ids, ray_cast_paths = bresenham_2d_batched(
                pose_idx[ray_env_ids], pts_idx_all, ray_env_ids, self.grid_size
            )
            self.prob_map.index_put_(
                (ray_cast_env_ids, ray_cast_paths[:, 0], ray_cast_paths[:, 1]),
                torch.full((ray_cast_paths.shape[0],), -0.05, device=self.device, dtype=self.prob_map.dtype),
                accumulate=True,
            )
            self.prob_map[ray_env_ids, pts_idx_all[:, 0], pts_idx_all[:, 1]] = 1.0
        occ_maps, self.occ_maps_tri_cls = discretize_prob_map(self.prob_map, threshold_occu=0.5, threshold_free=0.0)
        self.scanned_gt_map = torch.clip(self.scanned_gt_map + occ_maps * self.layout_maps_height_scenes, max=1, min=0)
        ego_prob_maps = extract_ego_maps(self.occ_maps_tri_cls.clone(), self.voxel_size_gt_scenes[:, :2], current_pose_idx, self.ego_cell_size)
        self.ego_prob_maps = ego_prob_maps.clone()
        ego_occ_masks = (ego_prob_maps != 1.0).to(torch.bool)
        ego_frontier_masks = compute_frontier_map(ego_prob_maps, self.frontier_kernel)
        ego_frontier_masks = ego_frontier_masks & ego_occ_masks
        self.ego_prob_maps[ego_frontier_masks] = 2.0

    def check_motion_collision_local_2d(self):
        cur_pose_idx = self.poses_idx.to(torch.long)
        self.collision_rigid = self.layout_maps_height_scenes[
            self.env_idx_tensor, cur_pose_idx[:, 0], cur_pose_idx[:, 1]
        ] == 1.0
        height_c = int(self.depth_processed_col.shape[1] / 2) - 1
        width_c = int(self.depth_processed_col.shape[2] / 2) - 1
        depth_center_area = torch.stack(
            [
                self.depth_processed_col[:, height_c, width_c],
                self.depth_processed_col[:, height_c, width_c + 1],
                self.depth_processed_col[:, height_c + 1, width_c],
                self.depth_processed_col[:, height_c + 1, width_c + 1],
            ],
            dim=1,
        ).abs()
        min_vis_dist, _ = torch.min(depth_center_area, dim=1)
        move_dist = torch.norm(self.move_dist, dim=-1)
        self.collision_vis = min_vis_dist < (move_dist - 0.15)
        self.collision_vis[self.cur_episode_length == 0] = False
        flag_no_need_local = torch.logical_or(self.collision_rigid, self.collision_vis == False)
        flag_no_need_local = torch.logical_or(flag_no_need_local, self.cur_episode_length == 0)
        env_mask = ~flag_no_need_local
        if torch.any(env_mask):
            occ_maps_bin_cls = 1.0 - (self.occ_maps_tri_cls[env_mask] >= 0.0).to(torch.float32)
            local_path = bfs_path_lengths(occ_maps_bin_cls, self.poses_idx_old[env_mask], self.poses_idx[env_mask])
            self.collision_vis[env_mask] = local_path == -1
        self.collision_flag = torch.logical_or(self.collision_rigid, self.collision_vis)

    def post_physics_step(self, if_reset: bool = False):
        self.episode_length_buf += 1
        obs, rewards, dones, infos = self.get_step_return()
        return obs if if_reset else (obs, rewards, dones, infos)

    def get_step_return(self):
        self.post_process_main_camera_tensor()
        self.update_observation()
        self.check_motion_collision_local_2d()
        self.compute_reward()
        obs = {
            "state": self.ego_pose_buf.view(self.num_envs, -1),
            "ego_map_2D": self.ego_prob_maps.reshape(self.num_envs, -1),
        }
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        rewards, dones, infos = self.rew_buf.clone(), self.reset_buf.clone(), self.extras
        self.update_extra_episode_info(rewards, dones)
        if env_ids.numel() > 0:
            self.reset_buf[env_ids] = 0
        return obs, rewards, dones, infos

    def update_extra_episode_info(self, rewards: torch.Tensor, dones: torch.Tensor):
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if new_ids.numel() > 0:
            self.rewbuffer.extend(self.cur_reward_sum[new_ids].detach().cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids].detach().cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0.0
            self.cur_episode_length[new_ids] = 0.0
        self.extras.setdefault("episode", {})
        self.extras["episode"]["episode_reward"] = (np.mean(self.rewbuffer) / self.max_episode_length_s) if self.rewbuffer else 0.0
        self.extras["episode"]["episode_length"] = np.mean(self.lenbuffer) if self.lenbuffer else 0.0

    def reset_idx(self, env_ids: torch.Tensor, switch_scene: bool = True):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self._debug(f"reset_idx:start env_ids={env_ids.tolist()}")
        changed_env_ids = []
        old_active_scene_ids = []
        new_active_scene_ids = []
        if switch_scene:
            for env_idx in env_ids.tolist():
                current_active = self.active_scene_ids[env_idx]
                if self.scene_per_env == 1:
                    next_scene = current_active
                else:
                    next_scene = (
                        (current_active - env_idx * self.scene_per_env + random.randint(1, self.scene_per_env - 1))
                        % self.scene_per_env
                    ) + env_idx * self.scene_per_env
                if next_scene != current_active:
                    changed_env_ids.append(env_idx)
                    old_active_scene_ids.append(current_active)
                    new_active_scene_ids.append(next_scene)
                    self.active_scene_ids[env_idx] = next_scene
        self._debug(f"reset_idx:new_active_scene_ids={new_active_scene_ids}")
        if changed_env_ids:
            changed_env_ids_tensor = torch.tensor(changed_env_ids, dtype=torch.long, device=self.device)
            new_active_scene_ids_tensor = torch.tensor(new_active_scene_ids, dtype=torch.long, device=self.device)
            old_active_scene_ids_tensor = torch.tensor(old_active_scene_ids, dtype=torch.long, device=self.device)
            self._debug("reset_idx:set_scene_positions_old")
            self._set_scene_positions(
                old_active_scene_ids_tensor,
                self.inactive_xyz.repeat(len(changed_env_ids), 1),
                self.scene_quat.repeat(len(changed_env_ids), 1),
            )
            self._debug("reset_idx:set_scene_positions_new")
            self._set_scene_positions(
                new_active_scene_ids_tensor,
                self.env_origins[changed_env_ids_tensor],
                self.scene_quat.repeat(len(changed_env_ids), 1),
            )
            self.env_to_scene[changed_env_ids_tensor] = new_active_scene_ids_tensor
            self.range_gt_scenes[changed_env_ids_tensor] = self.range_gt[new_active_scene_ids_tensor]
            self.voxel_size_gt_scenes[changed_env_ids_tensor] = self.voxel_size_gt[new_active_scene_ids_tensor]
            self.num_valid_pixel_gt_scenes[changed_env_ids_tensor] = self.num_valid_pixel_gt[new_active_scene_ids_tensor]
            self.layout_maps_height_scenes[changed_env_ids_tensor] = self.layout_maps_height[new_active_scene_ids_tensor]
        self._debug("reset_idx:sample_init_pose")
        self.ego_pose_buf[env_ids] = 0.0
        self._debug("reset_idx:ego_pose_buf:done")
        self.world_pose_buf[env_ids] = 0.0
        self._debug("reset_idx:world_pose_buf:done")
        for buf_idx in range(self.buffer_size):
            self.reward_layout_ratio_buf[buf_idx][env_ids] = 0.0
        self._debug("reset_idx:reward_layout_ratio_buf:done")
        self.actions[env_ids] = self.init_action_tensor
        self._debug("reset_idx:actions:done")
        for env_idx in env_ids.tolist():
            scene_idx = int(self.env_to_scene[env_idx].item())
            init_candidates = self.init_maps_list[scene_idx]
            self.poses[env_idx, :2] = init_candidates[random.randint(0, len(init_candidates) - 1)]
        self._debug("reset_idx:init_candidates:done")
        self.poses[:, 2] = self.motion_height
        self._debug("reset_idx:poses_height:done")
        self.prob_map[env_ids] = 0.0
        self._debug("reset_idx:prob_map:done")
        self.scanned_gt_map[env_ids] = 0.0
        self._debug("reset_idx:scanned_gt_map:done")
        self.occ_maps_tri_cls[env_ids] = 0.0
        self._debug("reset_idx:occ_maps_tri_cls:done")
        self.ego_prob_maps[env_ids] = 0.0
        self._debug("reset_idx:ego_prob_maps:done")
        self.episode_length_buf[env_ids] = 0
        self._debug("reset_idx:episode_length_buf:done")
        self.reset_buf[env_ids] = 1
        self._debug("reset_idx:reset_buf:done")
        self.extras["episode"] = {}
        self._debug("reset_idx:extras_episode:done")
        for key in self.episode_sums.keys():
            self.extras["episode"][f"rew_{key}"] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self._debug("reset_idx:episode_sums:done")
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            self._debug("reset_idx:timeouts:done")
        self._debug("reset_idx:done")

    def _set_scene_positions(self, scene_ids: torch.Tensor, positions: torch.Tensor, orientations: torch.Tensor):
        self.scene_view.set_world_poses(positions=positions, orientations=orientations, indices=scene_ids.tolist())

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        rew_surface = self._reward_surface_coverage_2d() * self.reward_scales["surface_coverage_2d"]
        rew_collision = self._reward_collision() * self.reward_scales["collision"]
        self.rew_buf += rew_surface + rew_collision
        self.episode_sums["surface_coverage_2d"] += rew_surface
        self.episode_sums["collision"] += rew_collision
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        self.check_termination()
        rew_term = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew_term
        self.episode_sums["termination"] += rew_term

    def check_termination(self):
        self.reset_buf = self.collision_flag.clone()
        recent_cr = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-self.recent_num]
        meaningless_wander = recent_cr < 0.01
        meaningless_wander *= self.cur_episode_length > self.recent_num
        self.reset_buf |= meaningless_wander.clone()
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf
        last_ratio = self.reward_layout_ratio_buf[-1]
        self.reset_buf |= last_ratio > self.ratio_threshold_term

    def _reward_surface_coverage_2d(self):
        layout_coverage = self.scanned_gt_map.sum(dim=(1, 2)) / self.num_valid_pixel_gt_scenes
        self.reward_layout_ratio_buf.extend([layout_coverage.clone()])
        rew_coverage = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-2]
        rew_coverage[self.collision_flag] = 0.0
        return rew_coverage

    def _reward_collision(self):
        return self.collision_flag.to(torch.float32)

    def _reward_termination(self):
        return self.reset_buf * (self.reward_layout_ratio_buf[-1] > self.ratio_threshold_rew)

    def render(self, mode="human"):
        return None

    def seed(self, seed: int | None = None):
        if seed is None:
            return [None] * self.num_envs
        self._set_random_seed(int(seed))
        return [int(seed) + env_idx for env_idx in range(self.num_envs)]

    def close(self):
        if self.sim is not None:
            self.sim._timeline.stop()
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self.sim = None
