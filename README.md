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

W&B 默认模式已经是 `offline`，所以命令中不需要额外写 `--wandb_mode=offline`。如果你要上传在线实验，再显式传 `--wandb_mode=online`。

### 4.1 Stage1（从零开始）

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless
```

### 4.2 Stage1（从 checkpoint 继续）

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

### 4.3 Stage2（基于 Stage1 checkpoint 继续）

`--ckpt_path` 必填：

```bash
python -m gleam_lab.train.train_stage2_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --total_iters=1250
```

训练步数计算：

- `total_timesteps = num_envs * n_steps * total_iters * 2`

## 5. Evaluation Command

迁移版评估入口：

```bash
python -m gleam_lab.test.test_gleam_gleambench_lab \
  --sim_device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip \
  --eval_episodes=128
```

评估输出路径：

- `runs_lab/eval_gleam_gleambench_lab_*/eval_summary.json`
- `runs_lab/eval_gleam_gleambench_lab_*/eval_episode_metrics.npz`

## 6. 参数说明（对应脚本 parse_args）

### 6.1 `train_stage1_lab.py`

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--device` | 自动推断（默认 `cuda:0`） | 训练与环境设备 |
| `--sim_device` | `None` | `--device` 的旧别名 |
| `--rl_device` | `None` | `--device` 的旧别名 |
| `--num_envs` | `32` | 并行环境数 |
| `--headless` | `False` | 无窗口模式 |
| `--backend` | `isaaclab` | 后端类型：`isaaclab` 或 `mock` |
| `--buffer_size` | `30` | 历史状态缓冲长度 |
| `--grid_size` | `128` | 地图网格边长 |
| `--n_steps` | `512` | PPO 每环境 rollout 步数 |
| `--batch_size` | `128` | PPO 小批量大小 |
| `--save_freq` | `50000` | 模型保存频率 |
| `--total_iters` | `1250` | 训练迭代预算 |
| `--n_epochs` | `5` | PPO 每轮优化 epoch 数 |
| `--use_target_kl` | `True` | 启用 KL 约束 |
| `--no_target_kl` | `False` | 禁用 KL 约束（与上面互斥） |
| `--target_kl` | `0.05` | KL 目标阈值 |
| `--vf_coeff` | `0.8` | value loss 系数 |
| `--ent_coeff` | `0.01` | entropy 系数 |
| `--lr` | `1e-4` | 学习率 |
| `--seed` | `1` | 随机种子 |
| `--exp_name` | `""` | 自定义实验名前缀 |
| `--ckpt_path` | `None` | 续训 checkpoint 路径（可选） |
| `--reset_num_timesteps` | `False` | 续训时是否把步数计数重置为 0 |
| `--stop_wandb` | `False` | 完全关闭 wandb 回调 |
| `--disable_progress_vis` | `False` | 关闭进度可视化图保存 |
| `--progress_vis_freq` | `50000` | 可视化快照频率 |
| `--progress_vis_env_id` | `0` | 可视化哪一个并行环境 |
| `--wandb_mode` | `offline` | wandb 模式：`online/offline/disabled` |

### 6.2 `train_stage2_lab.py`

除下面差异外，其余参数与 Stage1 一致：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--ckpt_path` | 必填 | Stage2 启动时必须提供 checkpoint |

### 6.3 `test_gleam_gleambench_lab.py`

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--device` | 自动推断（默认 `cuda:0`） | 评估设备 |
| `--sim_device` | `None` | `--device` 的旧别名 |
| `--rl_device` | `None` | `--device` 的旧别名 |
| `--num_envs` | `32` | 并行评估环境数 |
| `--headless` | `False` | 无窗口模式 |
| `--backend` | `isaaclab` | 后端类型：`isaaclab` 或 `mock` |
| `--buffer_size` | `30` | 历史状态缓冲长度 |
| `--grid_size` | `128` | 地图网格边长 |
| `--n_steps` | `512` | PPO 初始化参数（评估脚本会构建同结构模型） |
| `--batch_size` | `128` | PPO 初始化参数 |
| `--n_epochs` | `5` | PPO 初始化参数 |
| `--target_kl` | `0.05` | PPO 初始化参数 |
| `--vf_coeff` | `0.8` | PPO 初始化参数 |
| `--ent_coeff` | `0.01` | PPO 初始化参数 |
| `--lr` | `0.0` | 评估脚本默认不训练 |
| `--seed` | `0` | 随机种子 |
| `--eval_episodes` | `128` | 评估 episode 数 |
| `--exp_name` | `""` | 自定义评估实验名前缀 |
| `--deterministic` | `True` | 使用确定性策略评估 |
| `--ckpt_path` | 必填 | 待评估模型 checkpoint |
| `--stop_wandb` | `True` | 保留兼容参数，默认不走 wandb |

## 7. Legacy Commands (Original Isaac Gym Path)

Original upstream commands remain in `gleam/train/*.py` and `gleam/test/*.py`, but those paths depend on Isaac Gym + older CUDA/PyTorch assumptions and may fail on modern GPUs (for example `sm_120`).

For new hardware, prefer the `gleam_lab/` entries above.

## 8. Acknowledgement

This project is derived from the official GLEAM repository by the original authors.  
Please cite the original paper/repo if you use this codebase.
