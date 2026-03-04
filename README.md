# Lynx_gleam

`Lynx_gleam` is an Isaac-Lab-only fork of GLEAM for modern GPU stacks.

- Upstream: <https://github.com/zjwzcx/GLEAM>
- Paper: *GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scene* (ICCV 2025)
- This repo has removed legacy Isaac Gym paths and keeps only the Isaac Lab workflow under `gleam_lab/`.

## Highlights

- Isaac Lab training/evaluation entrypoints:
  - `python -m gleam_lab.train.train_stage1_lab ...`
  - `python -m gleam_lab.train.train_stage2_lab ...`
  - `python -m gleam_lab.test.test_gleam_gleambench_lab ...`
- W&B defaults to offline mode (`--wandb_mode=offline`).
- Supports `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`.

## Tested Stack

- Ubuntu 24.04
- Python 3.11
- PyTorch `2.7.0` (`cu128`)
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.2`

## Installation

```bash
bash gleam_lab/scripts/setup_gleam_lab_env.sh gleam_lab
conda activate gleam_lab
```

The setup script installs:
- CUDA 12.8 PyTorch stack
- Isaac Sim + Isaac Lab
- `gleam_lab/requirements_lab.txt`

## Dataset Layout

```text
data_gleam/
  train_stage1_512/
  train_stage2_512/
  eval_128/
```

If the folders above are missing, training/evaluation will fail.

## Training

### Stage1 (from scratch)

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless
```

### Stage1 (resume)

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

### Stage2

`--ckpt_path` is required:

```bash
python -m gleam_lab.train.train_stage2_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

Timesteps:

- `total_timesteps = num_envs * n_steps * total_iters * 2`

## Evaluation

```bash
python -m gleam_lab.test.test_gleam_gleambench_lab \
  --sim_device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --eval_episodes=128
```

Outputs:

- `runs_lab/eval_gleam_gleambench_lab_*/eval_summary.json`
- `runs_lab/eval_gleam_gleambench_lab_*/eval_episode_metrics.npz`

## Notes

- `--sim_device` and `--rl_device` are aliases of `--device`.
- `--backend=isaaclab` is default.
- Current Stage2 evaluator is migration-oriented (mock scaffold), not a 1:1 legacy Isaac Gym benchmark pipeline.

## Acknowledgement

This project is derived from the official GLEAM repository by the original authors.  
Please cite the original paper/repo if you use this codebase.
