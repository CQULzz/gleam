from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def compute_frontier_map(ego_prob_maps: torch.Tensor, frontier_kernel: torch.Tensor) -> torch.Tensor:
    ego_prob_maps_denoised = F.max_pool2d(ego_prob_maps.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    mask_neg1 = (ego_prob_maps_denoised == -1).float()
    mask_unknown = (ego_prob_maps_denoised == 0).float()
    unknown_neighbors = F.conv2d(mask_unknown, frontier_kernel, padding=1)
    frontier = (mask_neg1 == 1) & (unknown_neighbors >= 1)
    return frontier.squeeze(1)


def create_e2w_from_poses(poses: torch.Tensor, device: torch.device) -> torch.Tensor:
    num_env = poses.shape[0]
    x = poses[:, 0]
    y = poses[:, 1]
    yaw = poses[:, 5]
    cos_yaw = torch.cos(yaw).unsqueeze(1)
    sin_yaw = torch.sin(yaw).unsqueeze(1)
    return torch.cat(
        [
            torch.cat([cos_yaw, -sin_yaw, x.unsqueeze(1)], dim=1).unsqueeze(1),
            torch.cat([sin_yaw, cos_yaw, y.unsqueeze(1)], dim=1).unsqueeze(1),
            torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_env, 1, 3),
        ],
        dim=1,
    )


def scanned_pts_to_2d_idx(
    pts_target: torch.Tensor,
    range_gt_scenes: torch.Tensor,
    voxel_size_scenes: torch.Tensor,
    motion_height: float = 1.0,
    map_size: int = 256,
):
    num_env = pts_target.shape[0]
    motion_height_idx = ((motion_height - range_gt_scenes[:, 5]) / voxel_size_scenes[:, 2]).long()
    xyz_max_voxel = range_gt_scenes[:, [0, 2, 4]] + 0.5 * voxel_size_scenes
    xyz_min_voxel = range_gt_scenes[:, [1, 3, 5]] - 0.5 * voxel_size_scenes
    pts_target_idx = torch.floor((pts_target - xyz_min_voxel.unsqueeze(1)) / voxel_size_scenes.unsqueeze(1)).long()
    bound_mask = (xyz_max_voxel.unsqueeze(1) > pts_target) & (pts_target > xyz_min_voxel.unsqueeze(1))
    bound_mask = bound_mask.all(dim=-1)
    height_mask = pts_target_idx[..., 2] == motion_height_idx.unsqueeze(1)
    final_mask = bound_mask & height_mask

    pts_target_idxs = []
    for env_idx in range(num_env):
        valid_pts = pts_target_idx[env_idx][final_mask[env_idx]]
        if valid_pts.shape[0] == 0:
            pts_target_idxs.append([])
            continue
        valid_pts = torch.unique(valid_pts, dim=0)
        valid_pts = torch.clip(valid_pts, min=0, max=map_size - 1)
        pts_target_idxs.append(valid_pts[:, :2])
    return pts_target_idxs


def scanned_pts_to_2d_idx_batched(
    pts_target: torch.Tensor,
    range_gt_scenes: torch.Tensor,
    voxel_size_scenes: torch.Tensor,
    motion_height: float = 1.0,
    map_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_env = pts_target.shape[0]
    motion_height_idx = ((motion_height - range_gt_scenes[:, 5]) / voxel_size_scenes[:, 2]).long()
    xyz_max_voxel = range_gt_scenes[:, [0, 2, 4]] + 0.5 * voxel_size_scenes
    xyz_min_voxel = range_gt_scenes[:, [1, 3, 5]] - 0.5 * voxel_size_scenes
    pts_target_idx = torch.floor((pts_target - xyz_min_voxel.unsqueeze(1)) / voxel_size_scenes.unsqueeze(1)).long()
    bound_mask = (xyz_max_voxel.unsqueeze(1) > pts_target) & (pts_target > xyz_min_voxel.unsqueeze(1))
    bound_mask = bound_mask.all(dim=-1)
    height_mask = pts_target_idx[..., 2] == motion_height_idx.unsqueeze(1)
    final_mask = bound_mask & height_mask

    if not torch.any(final_mask):
        return (
            torch.empty((0,), device=pts_target.device, dtype=torch.long),
            torch.empty((0, 2), device=pts_target.device, dtype=torch.long),
        )

    env_ids = torch.arange(num_env, device=pts_target.device).unsqueeze(1).expand_as(final_mask)
    valid_env_ids = env_ids[final_mask]
    valid_pts = pts_target_idx[final_mask][:, :2]
    valid_pts = torch.clip(valid_pts, min=0, max=map_size - 1)
    valid_env_pts = torch.cat([valid_env_ids.unsqueeze(1), valid_pts], dim=1)
    valid_env_pts = torch.unique(valid_env_pts, dim=0)
    return valid_env_pts[:, 0].to(torch.long), valid_env_pts[:, 1:].to(torch.long)


def pose_coord_to_2d_idx(
    poses: torch.Tensor,
    range_gt_scenes: torch.Tensor,
    voxel_size_scenes: torch.Tensor,
    map_size: int = 256,
) -> torch.Tensor:
    x_min = range_gt_scenes[:, 1]
    y_min = range_gt_scenes[:, 3]
    voxel_sizes_xy = voxel_size_scenes[:, :2]
    xy_min_voxel = torch.stack([x_min, y_min], dim=-1) - 0.5 * voxel_sizes_xy
    if poses.dim() == 2:
        poses_idx = ((poses - xy_min_voxel) / voxel_sizes_xy).floor().long()
    elif poses.dim() == 3:
        poses_idx = ((poses - xy_min_voxel.unsqueeze(1)) / voxel_sizes_xy.unsqueeze(1)).floor().long()
    else:
        raise ValueError(f"Invalid poses shape: {poses.shape}")
    return torch.clip(poses_idx, min=0, max=map_size - 1)


def discretize_prob_map(
    grid_prob: torch.Tensor,
    threshold_occu: float = 0.5,
    threshold_free: float = 0.0,
):
    grid_occupancy = (grid_prob > threshold_occu).to(torch.float32)
    grid_free = (grid_prob < threshold_free).to(torch.float32)
    grid_tri_cls = grid_occupancy - grid_free
    return grid_occupancy, grid_tri_cls


def extract_ego_maps(global_maps: torch.Tensor, cell_sizes: torch.Tensor, poses_idx: torch.Tensor, ego_cm: float = 10.0):
    n, h, w = global_maps.shape
    device = global_maps.device
    h_cm = h * ego_cm
    w_cm = w * ego_cm
    patch_h_pixels = h_cm / cell_sizes[:, 0]
    patch_w_pixels = w_cm / cell_sizes[:, 1]
    half_patch_h = patch_h_pixels / 2.0
    half_patch_w = patch_w_pixels / 2.0
    t_y = torch.linspace(0, 1, steps=h, device=device).unsqueeze(0)
    t_x = torch.linspace(0, 1, steps=w, device=device).unsqueeze(0)
    start_y = poses_idx[:, 0].unsqueeze(1) - half_patch_h.unsqueeze(1)
    start_x = poses_idx[:, 1].unsqueeze(1) - half_patch_w.unsqueeze(1)
    y_coords = start_y + patch_h_pixels.unsqueeze(1) * t_y
    x_coords = start_x + patch_w_pixels.unsqueeze(1) * t_x
    grid_y = y_coords.unsqueeze(2).expand(-1, -1, w)
    grid_x = x_coords.unsqueeze(1).expand(-1, h, -1)
    norm_y = (grid_y / (h - 1)) * 2 - 1
    norm_x = (grid_x / (w - 1)) * 2 - 1
    batch_grid = torch.stack([norm_x, norm_y], dim=-1)
    ego_maps = F.grid_sample(
        input=global_maps.unsqueeze(1),
        grid=batch_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    return ego_maps.squeeze(1)


def bresenham_2d(pts_source: torch.Tensor, pts_target: torch.Tensor, map_size: int | Sequence[int]) -> torch.Tensor:
    """Torch fallback for the original PyCUDA Bresenham kernel.

    Returns the concatenated valid points across all rays, matching the original helper contract.
    """
    if isinstance(map_size, Sequence):
        map_size = int(map_size[0])
    tgt = pts_target.to(dtype=torch.int32)
    if tgt.numel() == 0:
        return torch.empty((0, 2), device=tgt.device, dtype=torch.long)

    device = tgt.device
    src = pts_source[0].to(dtype=torch.int32, device=device)
    num_rays = tgt.shape[0]
    max_pts_per_ray = int(map_size) * 2

    x = torch.full((num_rays,), int(src[0].item()), dtype=torch.int32, device=device)
    y = torch.full((num_rays,), int(src[1].item()), dtype=torch.int32, device=device)
    x1 = tgt[:, 0]
    y1 = tgt[:, 1]
    dx = torch.abs(x1 - x)
    dy = torch.abs(y1 - y)
    sx = torch.where(x < x1, 1, -1).to(torch.int32)
    sy = torch.where(y < y1, 1, -1).to(torch.int32)
    err = dx - dy

    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 2), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros((num_rays,), dtype=torch.int32, device=device)
    active = torch.ones((num_rays,), dtype=torch.bool, device=device)

    for _ in range(max_pts_per_ray):
        in_bounds = (x >= 0) & (x < map_size) & (y >= 0) & (y < map_size)
        write_mask = active & in_bounds
        if torch.any(write_mask):
            ray_idx = torch.nonzero(write_mask, as_tuple=False).squeeze(1)
            write_pos = trajectory_lengths[ray_idx].to(torch.long)
            trajectory_pts[ray_idx, write_pos, 0] = x[ray_idx]
            trajectory_pts[ray_idx, write_pos, 1] = y[ray_idx]
            trajectory_lengths[ray_idx] += 1

        reached = active & (x == x1) & (y == y1)
        active = active & ~reached
        if not torch.any(active):
            break

        e2 = 2 * err
        move_x = active & (e2 > -dy)
        move_y = active & (e2 < dx)

        err[move_x] -= dy[move_x]
        x[move_x] += sx[move_x]
        err[move_y] += dx[move_y]
        y[move_y] += sy[move_y]

    valid_mask = torch.arange(max_pts_per_ray, device=device).unsqueeze(0) < trajectory_lengths.unsqueeze(1)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, 2)
    return trajectory_pts[valid_mask].view(-1, 2).to(torch.long)


def bresenham_2d_batched(
    pts_source: torch.Tensor,
    pts_target: torch.Tensor,
    ray_env_ids: torch.Tensor,
    map_size: int | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Bresenham with per-ray env ids for direct index_put_ updates."""
    if isinstance(map_size, Sequence):
        map_size = int(map_size[0])
    tgt = pts_target.to(dtype=torch.int32)
    if tgt.numel() == 0:
        return (
            torch.empty((0,), device=tgt.device, dtype=torch.long),
            torch.empty((0, 2), device=tgt.device, dtype=torch.long),
        )

    device = tgt.device
    src = pts_source.to(dtype=torch.int32, device=device)
    if src.shape[0] != tgt.shape[0]:
        raise ValueError(f"Expected same number of sources and targets, got {src.shape[0]} and {tgt.shape[0]}")

    num_rays = tgt.shape[0]
    max_pts_per_ray = int(map_size) * 2

    x = src[:, 0].clone()
    y = src[:, 1].clone()
    x1 = tgt[:, 0]
    y1 = tgt[:, 1]
    dx = torch.abs(x1 - x)
    dy = torch.abs(y1 - y)
    sx = torch.where(x < x1, 1, -1).to(torch.int32)
    sy = torch.where(y < y1, 1, -1).to(torch.int32)
    err = dx - dy

    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 2), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros((num_rays,), dtype=torch.int32, device=device)
    active = torch.ones((num_rays,), dtype=torch.bool, device=device)

    for _ in range(max_pts_per_ray):
        in_bounds = (x >= 0) & (x < map_size) & (y >= 0) & (y < map_size)
        write_mask = active & in_bounds
        if torch.any(write_mask):
            ray_idx = torch.nonzero(write_mask, as_tuple=False).squeeze(1)
            write_pos = trajectory_lengths[ray_idx].to(torch.long)
            trajectory_pts[ray_idx, write_pos, 0] = x[ray_idx]
            trajectory_pts[ray_idx, write_pos, 1] = y[ray_idx]
            trajectory_lengths[ray_idx] += 1

        reached = active & (x == x1) & (y == y1)
        active = active & ~reached
        if not torch.any(active):
            break

        e2 = 2 * err
        move_x = active & (e2 > -dy)
        move_y = active & (e2 < dx)

        err[move_x] -= dy[move_x]
        x[move_x] += sx[move_x]
        err[move_y] += dx[move_y]
        y[move_y] += sy[move_y]

    point_mask = torch.arange(max_pts_per_ray, device=device).unsqueeze(0) < trajectory_lengths.unsqueeze(1)
    flat_pts = trajectory_pts[point_mask].view(-1, 2).to(torch.long)
    flat_env_ids = ray_env_ids.to(torch.long).unsqueeze(1).expand(-1, max_pts_per_ray)[point_mask]
    return flat_env_ids, flat_pts


def quat_wxyz_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def rotmat_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def transform_from_pose(position: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = rotmat_from_quat_wxyz(quat_wxyz)
    batch = position.shape[0]
    transform = torch.eye(4, device=position.device, dtype=position.dtype).unsqueeze(0).repeat(batch, 1, 1)
    transform[:, :3, :3] = rot
    transform[:, :3, 3] = position
    return transform


def build_env_origins(num_envs: int, spacing: float, device: torch.device) -> torch.Tensor:
    cols = math.ceil(math.sqrt(num_envs))
    origins = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    for env_id in range(num_envs):
        row = env_id // cols
        col = env_id % cols
        origins[env_id, 0] = row * spacing
        origins[env_id, 1] = col * spacing
    return origins
