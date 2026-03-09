# GLEAM Lab Migration

This folder currently contains the completed and validated Stage1 migration path for running GLEAM on modern GPUs (e.g. `sm_120`) with a new PyTorch/CUDA stack.

## Quick Start

1. Create a new environment and install Isaac Lab stack (see `gleam_lab/scripts/setup_gleam_lab_env.sh`).
   The script uses `gleam_lab/requirements_lab.txt` and intentionally avoids the legacy root `requirements.txt`.
2. Run Stage-1 migration entry:

```bash
python -m gleam_lab.train.train_stage1_lab --device=cuda:0 --num_envs=32 --headless
```

Progress images are saved by default to `runs_lab/<trial_name>/progress/latest.png` and
`runs_lab/<trial_name>/progress/progress_step_*.png`.

## Continue Training

If a run reaches the configured timestep budget, resume from a checkpoint:

```bash
python -m gleam_lab.train.train_stage1_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --wandb_mode=offline \
  --ckpt_path runs_lab/<trial_name>/models/rl_model_40960000_steps.zip \
  --total_iters=1250
```

- With `--ckpt_path`, the default behavior is to **continue** timestep counting (`reset_num_timesteps=False`).
- Add `--reset_num_timesteps` only if you want to reset counters to 0.
- `Early stopping at step ... due to reaching max kl` is PPO epoch early-stop, not process exit.
- Use `--no_target_kl` to disable KL-based early-stop inside PPO update.

## Stage2 Training

Start Stage2 from a Stage1 (or previous Stage2) checkpoint:

```bash
python -m gleam_lab.train.train_stage2_lab \
  --device=cuda:0 \
  --num_envs=32 \
  --headless \
  --wandb_mode=offline \
  --ckpt_path runs_lab/<stage1_trial>/models/rl_model_40960000_steps.zip \
  --total_iters=1250
```

- `--ckpt_path` is required for Stage2.
- Stage2 entry keeps the same PPO stack/CLI semantics as Stage1 migration path.
- Current Stage2 migration is not completed or validated yet.
- Do not treat Stage2 as a finished migration target.

## Benchmark-Style Eval

Run a checkpoint evaluation with the migration stack:

```bash
python -m gleam_lab.test.test_gleam_gleambench_lab \
  --sim_device=cuda:0 \
  --num_envs=32 \
  --headless \
  --stop_wandb \
  --ckpt_path /abs/path/to/rl_model_xxx_steps.zip
```

- This CLI mirrors the original `test_gleam_gleambench.py` argument style.
- Results are written to `runs_lab/eval_gleam_gleambench_lab_*/eval_summary.json`.
- This evaluator is not yet a completed or validated replacement for the legacy Isaac Gym `eval_128` benchmark implementation.

## Current Scope

- Completed and validated: Stage1 migration
- Not completed / not validated yet: Stage2 migration, benchmark-style evaluator

## Notes

- `--sim_device` and `--rl_device` are accepted as aliases of `--device`.
- `--backend=isaaclab` is default and tries to launch Isaac Lab app context.
- `WANDB_API_KEY` is now preferred. If missing, the callback falls back to `wandb_utils/wandb_api_key_file.txt`.

## Smoke Test

Use this to quickly validate training loop and callbacks:

```bash
python -m gleam_lab.train.train_stage1_lab \
  --backend=mock \
  --device=cpu \
  --num_envs=2 \
  --n_steps=16 \
  --total_iters=2 \
  --stop_wandb \
  --headless
```
