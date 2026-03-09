from __future__ import annotations

import argparse
import os
import time

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy_Discrete
from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
from stable_baselines3.utils import get_time_str

from gleam_lab.app import launch_app
OPEN_ROBOT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLEAM Stage1 training on Isaac Lab")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--stop_wandb", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--eval_device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--total_iters", type=int, default=1250)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--use_target_kl", type=bool, default=True)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--vf_coeff", type=float, default=0.8)
    parser.add_argument("--ent_coeff", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--surface_coverage", type=float, default=1.0)
    parser.add_argument("--only_positive_rewards", type=bool, default=False)
    parser.add_argument("--max_contact_force", type=float, default=100.0)
    parser.add_argument("--dataset_name", type=str, default="stage1_512")
    parser.add_argument("--data_root", type=str, default=os.path.join(OPEN_ROBOT_ROOT_DIR, "data_gleam"))
    parser.add_argument("--num_scene_override", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_app = launch_app(headless=args.headless, enable_cameras=True)

    from gleam.network.encoder import Encoder_GLEAM
    from gleam_lab.callback import ReconstructionCallBack
    from gleam_lab.env.config_gleam_lab import build_stage1_lab_config
    from gleam_lab.env.env_gleam_stage1_lab import Env_GLEAM_Stage1_Lab
    from gleam_lab.wrapper import EnvWrapperGLEAMLab

    project_name = "gleam"
    team_name = None
    WandbCallback = None

    try:
        if not args.stop_wandb:
            from wandb_utils import project_name as wandb_project_name
            from wandb_utils.wandb_callback import WandbCallback as _WandbCallback

            project_name = wandb_project_name
            team_name = getattr(__import__("wandb_utils"), "team_name", None)
            WandbCallback = _WandbCallback

        exp_name = "train_gleam_stage1_lab"
        trial_name = f"{exp_name}_{get_time_str()}" if not args.exp_name else f"{args.exp_name}_{get_time_str()}"
        log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs_lab", trial_name)
        os.makedirs(log_dir, exist_ok=True)

        env_cfg = build_stage1_lab_config(args)
        env_train = Env_GLEAM_Stage1_Lab(env_cfg)
        env_cfg_dict = vars(args).copy()
        env = EnvWrapperGLEAMLab(env_train)

        config = dict(
            algo=dict(
                policy=ActorCriticPolicy_Discrete,
                policy_kwargs=dict(
                    net_arch=[],
                    features_extractor_class=Encoder_GLEAM,
                    features_extractor_kwargs=dict(
                        encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
                        net_param={"transformer_params": [[1, 256], [1, 256]], "append_hidden_shapes": [256, 256]},
                        state_input_shape=(args.buffer_size * 6,),
                        visual_input_shape=(1, 128, 128),
                    ),
                ),
                env=env,
                learning_rate=args.lr,
                gamma=0.99,
                gae_lambda=0.95,
                target_kl=args.target_kl if args.use_target_kl else None,
                max_grad_norm=1,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                clip_range=0.2,
                vf_coef=args.vf_coeff,
                clip_range_vf=0.2,
                ent_coef=args.ent_coeff,
                tensorboard_log=log_dir,
                create_eval_env=False,
                verbose=2,
                seed=args.seed,
                device=args.sim_device,
            ),
            gpu_simulation=True,
            project_name=project_name,
            team_name=team_name,
            exp_name=exp_name,
            seed=args.seed,
            use_wandb=not args.stop_wandb,
            trial_name=trial_name,
            log_dir=log_dir,
        )

        callbacks = [
            ReconstructionCallBack(
                name_prefix="rl_model",
                verbose=1,
                save_freq=args.save_freq,
                save_path=os.path.join(log_dir, "models"),
                key_list=["episode_reward"],
            )
        ]
        if not args.stop_wandb:
            callbacks.append(WandbCallback(trial_name=trial_name, exp_name=exp_name, project_name=project_name, config={**config, **env_cfg_dict}))
        callbacks = CallbackList(callbacks)

        model = PPO_Grid_Obs(**config["algo"])
        model.learn(
            total_timesteps=args.num_envs * args.n_steps * args.total_iters * 2,
            callback=callbacks,
            reset_num_timesteps=True,
            eval_env=None,
            tb_log_name=exp_name,
            log_interval=1,
        )
    finally:
        try:
            env.close()  # type: ignore[name-defined]
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()
    main()
    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Total wall-clock time: {:.3f}min".format((t_end - t_start) / 60))
