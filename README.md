# GLEAM (Isaac Lab Migration)

This repository is based on the official GLEAM project:

- Upstream code: <https://github.com/zjwzcx/GLEAM>
- Paper: *GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scene* (ICCV 2025)

This fork keeps original GLEAM code paths and adds a parallel `gleam_lab/` path for modern GPU compatibility (for example RTX 50 series / `sm_120`), using Isaac Lab instead of legacy Isaac Gym for day-to-day training/evaluation workflows.

## 1. What Is Different In This Repo

- Keep original folders (`gleam/`, `legged_gym/`, etc.) for reference.
- Add migration path under `gleam_lab/`:
  - Stage1 training entry: `python -m gleam_lab.train.train_stage1_lab ...`
  - Stage2 training entry: `python -m gleam_lab.train.train_stage2_lab ...`
  - Benchmark-style eval entry: `python -m gleam_lab.test.test_gleam_gleambench_lab ...`
- W&B behavior:
  - default `wandb_mode=offline`
  - `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT` env vars are supported
  - key file fallback remains for compatibility

## 2. Environment (Isaac Lab Stack)

Tested stack in this repo:

- Ubuntu 24.04
- Python 3.11
- PyTorch `2.7.0` (`cu128`)
- torchvision `0.22.0` (`cu128`)
- torchaudio `2.7.0` (`cu128`)
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.2`
- numpy `1.26.0`
- opencv-python `4.11.0.86`
- wandb `0.25.0`

Install with one command:

```bash
bash gleam_lab/scripts/setup_gleam_lab_env.sh gleam_lab
conda activate gleam_lab
```

The setup script is here: [setup_gleam_lab_env.sh](/home/lzz/GLEAM/gleam_lab/scripts/setup_gleam_lab_env.sh)

## 3. Data Layout

Expected folders:

```text
data_gleam/
  train_stage1_512/
  train_stage2_512/
  eval_128/
```

If these datasets are missing, training/evaluation commands that depend on them will fail.

## 4. Training Commands

### 4.1 Stage1 (from scratch)

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --wandb_mode=offline
```

### 4.2 Stage1 (resume from checkpoint)

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --wandb_mode=offline \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

### 4.3 Stage2 (continue from Stage1 checkpoint)

`--ckpt_path` is required:

```bash
python -m gleam_lab.train.train_stage2_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --wandb_mode=offline \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

Notes:

- `total_timesteps = num_envs * n_steps * total_iters * 2`
- If you set `--total_iters=2500`, training budget/time is roughly doubled.
- PPO log `Early stopping at step ... due to reaching max kl` is epoch-level early stop inside one update, not process crash.

## 5. Evaluation Command

Migration benchmark-style eval entry:

```bash
python -m gleam_lab.test.test_gleam_gleambench_lab \
  --sim_device=cuda:0 \
  --num_envs=32 \
  --headless \
  --stop_wandb \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --eval_episodes=128
```

Outputs are saved under:

- `runs_lab/eval_gleam_gleambench_lab_*/eval_summary.json`
- `runs_lab/eval_gleam_gleambench_lab_*/eval_episode_metrics.npz`

## 6. Legacy Commands (Original Isaac Gym Path)

Original upstream commands remain in `gleam/train/*.py` and `gleam/test/*.py`, but those paths depend on Isaac Gym + older CUDA/PyTorch assumptions and may fail on modern GPUs (for example `sm_120`).

For new hardware, prefer the `gleam_lab/` entries above.

## 7. Acknowledgement

This project is derived from the official GLEAM repository by the original authors.  
Please cite the original paper/repo if you use this codebase.

