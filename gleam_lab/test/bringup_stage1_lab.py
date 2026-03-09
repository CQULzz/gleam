from __future__ import annotations

import argparse

import torch

from gleam_lab.app import launch_app
from gleam_lab.env.config_gleam_lab import ConfigGLEAMLabStage1


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 Isaac Lab bring-up")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--num_scene_override", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_app = launch_app(headless=args.headless, enable_cameras=True)
    from gleam_lab.env.env_gleam_stage1_lab import Env_GLEAM_Stage1_Lab

    try:
        cfg = ConfigGLEAMLabStage1(headless=args.headless, device=args.sim_device, num_scene_override=args.num_scene_override)
        cfg.env.num_envs = args.num_envs
        cfg.visual_input.stack = 30
        env = Env_GLEAM_Stage1_Lab(cfg)
        obs = env.reset()
        print("reset/state", tuple(obs["state"].shape), "reset/map", tuple(obs["ego_map_2D"].shape))
        action = torch.tensor(cfg.normalization.init_action, device=env.device).repeat(env.num_envs, 1)
        obs, rew, done, info = env.step(action)
        print("step/reward", rew.detach().cpu().tolist())
        print("step/done", done.detach().cpu().tolist())
        print("episode keys", sorted(info.get("episode", {}).keys()))
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
