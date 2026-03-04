import argparse
from dataclasses import dataclass

from gleam_lab.env.stage1_lab_env import GLEAMStage1MockEnv, Stage1LabEnvConfig


@dataclass
class Stage2LabEnvConfig(Stage1LabEnvConfig):
    """Stage2 migration config.

    Kept compatible with Stage1 migration scaffold for now.
    """


class GLEAMStage2MockEnv(GLEAMStage1MockEnv):
    """Stage2 migration scaffold env.

    Currently reuses Stage1 tensorized mock dynamics until full Stage2 Isaac Lab env port.
    """


def build_stage2_lab_config(args: argparse.Namespace) -> Stage2LabEnvConfig:
    return Stage2LabEnvConfig(
        num_envs=int(args.num_envs),
        device=str(args.device),
        headless=bool(args.headless),
        backend=str(args.backend),
        buffer_size=int(args.buffer_size),
        grid_size=int(args.grid_size),
    )
