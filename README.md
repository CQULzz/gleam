<p align="center">
<h1 align="center"><strong>GLEAM</strong></h1>
<p align="center">
Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes<br>
Current fork with an Isaac Lab migration path for modern RTX GPUs.
</p>
</p>

- Upstream: <https://github.com/zjwzcx/GLEAM>
- Paper: *GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scene* (ICCV 2025)
- This repository keeps the original upstream `gleam/` codebase and adds a runnable Isaac Lab migration under `gleam_lab/`.

## Repo Status

This fork has two code paths:

1. `gleam/`
- Original upstream Isaac Gym implementation.
- Kept for reference and source parity.
- Requires the legacy Isaac Gym stack and is not the recommended path on RTX 50-series GPUs.

2. `gleam_lab/`
- Active migration path for modern stacks.
- Only Stage1 has been completed and validated at this time.
- Keeps the original Stage1 logic as closely as practical:
  - online multi-camera depth sensing
  - occupancy / frontier update
  - reward / termination structure
  - PPO / encoder / policy path
  - original `bfs_cuda_2D` CUDA extension

## Tested Stack

Validated on:

- Ubuntu 24.04
- Python 3.11
- PyTorch 2.7.0 + CUDA 12.8
- Isaac Sim 5.1.0.0
- Isaac Lab 2.3.2
- NVIDIA RTX 5060 Ti

Recommended conda env name:

```bash
conda activate gleam_lab_clean
```

## Dataset Layout

```text
data_gleam/
  train_stage1_512/
  train_stage2_512/
  eval_128/
```

If these folders are missing, training and evaluation will fail.

## Installation

### Modern Isaac Lab path

Install the runtime dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Build the CUDA BFS extension in place:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py build_ext --inplace
```

### Legacy Isaac Gym path

For the original `gleam/` path, follow the upstream environment instructions from the official repository. This fork does not claim to modernize that legacy runtime.

## Training

### Stage1 on Isaac Lab

This is the only migrated path that is currently completed and validated.

Validated commands:

```bash
python -m gleam_lab.train.train_stage1_lab \
  --headless \
  --stop_wandb \
  --sim_device cuda:0 \
  --num_envs 4 \
  --num_scene_override 16
```

Medium-scale run:

```bash
python -m gleam_lab.train.train_stage1_lab \
  --headless \
  --stop_wandb \
  --sim_device cuda:0 \
  --num_envs 8 \
  --num_scene_override 64
```

Logs are written to `runs_lab/`.

### Original upstream Stage1

```bash
python gleam/train/train_gleam_stage1.py --sim_device=cuda:0 --num_envs=32 --headless
```

### Stage2

- `train_stage2_lab.py` exists in the repository, but Stage2 has not been completed or validated in the Isaac Lab migration path.
- Do not treat Stage2 as a finished migration target yet.

## Evaluation

The migrated evaluator entry exists in the repository:

```bash
python -m gleam_lab.test.test_gleam_gleambench_lab \
  --sim_device=cuda:0 \
  --num_envs=32 \
  --headless \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip
```

Outputs:

- `runs_lab/eval_gleam_gleambench_lab_*/eval_summary.json`
- `runs_lab/eval_gleam_gleambench_lab_*/eval_episode_metrics.npz`

Current status:

- The repository currently only guarantees a completed and validated Stage1 migration path.
- Stage2 and the migrated evaluator are still unfinished / unvalidated and should be treated as work in progress.

## Notes

- This fork does not use the earlier mock / reveal-disk shortcut for the validated Stage1 path.
- The Isaac Lab path restores the original GPU BFS extension and keeps online depth-based map updates.
- Current optimization work focuses on making the Isaac Lab Stage1 path faster and more scalable without changing the original training logic.

## Citation

If you use this repository, please cite the original GLEAM paper:

```bibtex
@article{chen2025gleam,
  title={GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes},
  author={Chen, Xiao and Wang, Tai and Li, Quanyi and Huang, Tao and Pang, Jiangmiao and Xue, Tianfan},
  journal={arXiv preprint arXiv:2505.20294},
  year={2025}
}
```
