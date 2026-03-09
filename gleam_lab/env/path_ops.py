from __future__ import annotations

from collections import deque

import torch

try:
    import bfs_cuda_2D
except ImportError:
    bfs_cuda_2D = None


def _bfs_path_lengths_cpu(occupancy_maps: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """CPU fallback matching the original extension contract.

    Input semantics match the original extension call: 1 indicates free, 0 indicates blocked.
    Returns -1 when no path exists. Path lengths are 1-based to match the original CUDA kernel.
    """
    occ_cpu = occupancy_maps.detach().to(dtype=torch.int32, device="cpu")
    starts_cpu = starts.detach().to(dtype=torch.int32, device="cpu")
    goals_cpu = goals.detach().to(dtype=torch.int32, device="cpu")
    out = torch.full((occupancy_maps.shape[0],), -1.0, dtype=torch.float32, device=occupancy_maps.device)

    for env_id in range(occ_cpu.shape[0]):
        grid = occ_cpu[env_id]
        start = tuple(int(v) for v in starts_cpu[env_id].tolist())
        goal = tuple(int(v) for v in goals_cpu[env_id].tolist())
        if grid[start[0], start[1]] == 0 or grid[goal[0], goal[1]] == 0:
            continue
        if start == goal:
            out[env_id] = 1.0
            continue
        q = deque([(start[0], start[1], 1)])
        visited = {start}
        found = -1
        while q:
            x, y, dist = q.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= grid.shape[0] or ny >= grid.shape[1]:
                    continue
                if grid[nx, ny] == 0:
                    continue
                if (nx, ny) in visited:
                    continue
                if (nx, ny) == goal:
                    found = dist + 1
                    q.clear()
                    break
                visited.add((nx, ny))
                q.append((nx, ny, dist + 1))
        if found >= 0:
            out[env_id] = float(found)
    return out


def bfs_path_lengths(occupancy_maps: torch.Tensor, starts: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Return original GLEAM 4-neighbor shortest-path lengths on a binary occupancy grid."""
    if bfs_cuda_2D is not None and occupancy_maps.is_cuda and starts.is_cuda and goals.is_cuda:
        out = torch.full((occupancy_maps.shape[0],), -1.0, dtype=torch.float32, device=occupancy_maps.device)
        bfs_cuda_2D.BFS_CUDA_2D(
            occupancy_maps.contiguous().to(dtype=torch.float32),
            starts.contiguous().to(dtype=torch.int32),
            goals.contiguous().to(dtype=torch.int32),
            out,
        )
        return out
    return _bfs_path_lengths_cpu(occupancy_maps, starts, goals)
