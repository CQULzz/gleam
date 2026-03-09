import os
from typing import List

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

from stable_baselines3.common.callbacks import BaseCallback

# Headless-safe backend for saving figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ProgressVisCallback(BaseCallback):
    """Save training progress snapshots as PNG files."""

    def __init__(
        self,
        save_path: str,
        save_freq: int = 50_000,
        env_id: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.save_freq = int(save_freq)
        self.env_id = int(env_id)
        self._last_saved_timestep = 0
        self._history_steps: List[int] = []
        self._history_coverage: List[float] = []

        # -1 unknown -> dark gray, 0 free -> light gray, 1 occupied/explored -> green, 2 frontier -> orange
        self._cmap = ListedColormap(["#2f2f2f", "#bdbdbd", "#4caf50", "#ff9800"])

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        # Always save an initial snapshot.
        self._save_snapshot(force=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved_timestep >= self.save_freq:
            self._save_snapshot(force=False)
        return True

    def _on_training_end(self) -> None:
        self._save_snapshot(force=True)

    def _save_snapshot(self, force: bool) -> None:
        env = self.training_env
        if env is None or not hasattr(env, "_ego_map") or not hasattr(env, "_coverage_ratio"):
            return
        if self.env_id < 0 or self.env_id >= int(getattr(env, "num_envs", 1)):
            return

        step = int(self.num_timesteps)
        if (not force) and step == self._last_saved_timestep:
            return
        self._last_saved_timestep = step

        ego_map = env._ego_map[self.env_id].detach().cpu().numpy()
        coverage = float(env._coverage_ratio[self.env_id].detach().cpu().item())
        self._history_steps.append(step)
        self._history_coverage.append(coverage)

        # Shift values to [0, 3] for colormap indexing.
        vis_map = np.clip((ego_map + 1.0).astype(np.int32), 0, 3)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
        ax_map, ax_curve = axes

        im = ax_map.imshow(vis_map, cmap=self._cmap, vmin=0, vmax=3)
        ax_map.set_title(f"Env {self.env_id} Map @ step {step:,}\ncoverage={coverage:.3f}")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["unknown", "free", "explored", "frontier"])

        ax_curve.plot(self._history_steps, self._history_coverage, color="#1976d2", linewidth=2)
        ax_curve.set_title("Coverage Progress")
        ax_curve.set_xlabel("Timesteps")
        ax_curve.set_ylabel("Coverage Ratio")
        ax_curve.set_ylim(0.0, 1.0)
        ax_curve.grid(True, alpha=0.3)

        fig.tight_layout()
        out_file = os.path.join(self.save_path, f"progress_step_{step:010d}_env{self.env_id}.png")
        fig.savefig(out_file)
        plt.close(fig)

        latest_file = os.path.join(self.save_path, "latest.png")
        try:
            # Replace by copy-like overwrite to keep portable behavior.
            from shutil import copyfile

            copyfile(out_file, latest_file)
        except Exception:
            pass

        if self.verbose > 0:
            print(f"[PROGRESS_VIS] Saved {out_file}")
