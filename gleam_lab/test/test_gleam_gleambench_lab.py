import argparse
import json
import os
import time

import numpy as np
import torch

from gleam_lab.network.encoder import Encoder_GLEAM
from gleam_lab.env import GLEAMStage2MockEnv, build_stage2_lab_config
from gleam_lab.wrapper import EnvWrapperGLEAMLab
from stable_baselines3.common.policies import ActorCriticPolicy_Discrete
from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
from stable_baselines3.utils import get_time_str

OPEN_ROBOT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="GLEAM benchmark-style evaluation entrypoint (Isaac Lab migration stack).")
    parser.add_argument("--device", type=str, default=None, help="RL and env device, e.g. cuda:0 or cpu")
    parser.add_argument("--sim_device", type=str, default=None, help="Legacy alias for --device")
    parser.add_argument("--rl_device", type=str, default=None, help="Legacy alias for --device")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="isaaclab", choices=["isaaclab", "mock"])
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--vf_coeff", type=float, default=0.8)
    parser.add_argument("--ent_coeff", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.0, help="Set to 0.0 for pure evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=128)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to a Stage1/Stage2 saved *.zip checkpoint.",
    )
    parser.add_argument("--stop_wandb", action="store_true", default=True, help="Reserved for CLI compatibility.")
    args = parser.parse_args()

    if args.device is None:
        args.device = args.sim_device or args.rl_device or "cuda:0"
    return args


def _to_tensor(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def evaluate_policy(model, env, eval_episodes: int, deterministic: bool = True):
    """Simple vectorized evaluation loop for migration stack."""
    obs = env.reset()
    device = env.device if hasattr(env, "device") else torch.device("cpu")

    ep_rewards = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    ep_lengths = torch.zeros(env.num_envs, dtype=torch.int64, device=device)
    done_rewards = []
    done_lengths = []
    done_coverages = []

    while len(done_rewards) < eval_episodes:
        actions, _ = model.predict(obs, state=None, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)
        rewards_t = _to_tensor(rewards, device=device).float()
        dones_t = _to_tensor(dones, device=device).bool()

        ep_rewards += rewards_t
        ep_lengths += 1

        done_ids = torch.nonzero(dones_t, as_tuple=False).flatten().tolist()
        if not done_ids:
            continue

        # Current mock env reports only a mean coverage for done envs in this step.
        coverage_scalar = 0.0
        if isinstance(infos, dict):
            episode_info = infos.get("episode", {})
            if isinstance(episode_info, dict) and "coverage_ratio" in episode_info:
                coverage_scalar = float(_to_tensor(episode_info["coverage_ratio"], device=device).mean().item())

        for env_id in done_ids:
            if len(done_rewards) >= eval_episodes:
                break
            done_rewards.append(float(ep_rewards[env_id].item()))
            done_lengths.append(int(ep_lengths[env_id].item()))
            done_coverages.append(coverage_scalar)
            ep_rewards[env_id] = 0.0
            ep_lengths[env_id] = 0

    rewards_np = np.asarray(done_rewards, dtype=np.float32)
    lengths_np = np.asarray(done_lengths, dtype=np.int32)
    coverage_np = np.asarray(done_coverages, dtype=np.float32)
    return dict(
        episode_rewards=done_rewards,
        episode_lengths=done_lengths,
        episode_coverages=done_coverages,
        mean_reward=float(rewards_np.mean()),
        std_reward=float(rewards_np.std()),
        mean_ep_length=float(lengths_np.mean()),
        std_ep_length=float(lengths_np.std()),
        mean_coverage=float(coverage_np.mean()),
        std_coverage=float(coverage_np.std()),
    )


def main():
    args = parse_args()

    ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--ckpt_path does not exist: {ckpt_path}")

    exp_name = "eval_gleam_gleambench_lab"
    trial_name = f"{exp_name}_{get_time_str()}" if len(args.exp_name) == 0 else f"{args.exp_name}_{get_time_str()}"
    log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs_lab", trial_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOGGING] Start evaluation logging into {log_dir}")
    print(f"[EVAL] Loading checkpoint from {ckpt_path}")

    env_cfg = build_stage2_lab_config(args)
    env = EnvWrapperGLEAMLab(GLEAMStage2MockEnv(env_cfg))

    config = dict(
        policy=ActorCriticPolicy_Discrete,
        policy_kwargs=dict(
            net_arch=[],
            features_extractor_class=Encoder_GLEAM,
            features_extractor_kwargs=dict(
                encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
                net_param={
                    "transformer_params": [[1, 256], [1, 256]],
                    "append_hidden_shapes": [256, 256],
                },
                state_input_shape=(args.buffer_size * 6,),
                visual_input_shape=(1, args.grid_size, args.grid_size),
            ),
        ),
        env=env,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=args.target_kl,
        max_grad_norm=1.0,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=0.2,
        vf_coef=args.vf_coeff,
        clip_range_vf=0.2,
        ent_coef=args.ent_coeff,
        tensorboard_log=log_dir,
        create_eval_env=False,
        verbose=1,
        seed=int(args.seed),
        device=args.device,
    )

    model = PPO_Grid_Obs(**config)
    model.set_parameters(ckpt_path, exact_match=False, device=args.device)

    results = evaluate_policy(
        model=model,
        env=env,
        eval_episodes=int(args.eval_episodes),
        deterministic=bool(args.deterministic),
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "ckpt_path": ckpt_path,
        "eval_episodes": int(args.eval_episodes),
        "num_envs": int(args.num_envs),
        "device": str(args.device),
        "backend": str(args.backend),
        "mean_reward": results["mean_reward"],
        "std_reward": results["std_reward"],
        "mean_ep_length": results["mean_ep_length"],
        "std_ep_length": results["std_ep_length"],
        "mean_coverage": results["mean_coverage"],
        "std_coverage": results["std_coverage"],
    }

    summary_path = os.path.join(log_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez(
        os.path.join(log_dir, "eval_episode_metrics.npz"),
        episode_rewards=np.asarray(results["episode_rewards"], dtype=np.float32),
        episode_lengths=np.asarray(results["episode_lengths"], dtype=np.int32),
        episode_coverages=np.asarray(results["episode_coverages"], dtype=np.float32),
    )

    print("[EVAL] Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[EVAL] Saved summary to {summary_path}")
    print(
        "[EVAL][NOTE] This is the Isaac-Lab migration evaluator on the current Stage2 mock scaffold, "
        "not the legacy Isaac Gym eval_128 benchmark pipeline."
    )
    # In current Isaac Sim/Lab runtime, app.close() may terminate the process immediately.
    # Keep it at the very end so summary files are already flushed to disk.
    try:
        env.close()
    except Exception as exc:
        print(f"[EVAL][WARN] env.close() failed: {exc}")


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()
    main()
    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"Total wall-clock time: {(t_end - t_start) / 60:.3f}min")
