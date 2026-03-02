import argparse
import os
import time

from gleam.callback import ReconstructionCallBack
from gleam.network.encoder import Encoder_GLEAM
from gleam_lab.callbacks import ProgressVisCallback
from gleam_lab.env import GLEAMStage1MockEnv, build_stage1_lab_config
from gleam_lab.wrapper import EnvWrapperGLEAMLab
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy_Discrete
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
from stable_baselines3.utils import get_time_str
from wandb_utils import project_name as default_project_name
from wandb_utils import team_name as default_team_name
from wandb_utils.wandb_callback import WandbCallback

OPEN_ROBOT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="GLEAM Stage1 migration entrypoint (Isaac Lab stack).")
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
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--total_iters", type=int, default=1250)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--use_target_kl", dest="use_target_kl", action="store_true")
    parser.add_argument("--no_target_kl", dest="use_target_kl", action="store_false")
    parser.set_defaults(use_target_kl=True)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--vf_coeff", type=float, default=0.8)
    parser.add_argument("--ent_coeff", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional path to a saved *.zip checkpoint. When set, training resumes from this model.",
    )
    parser.add_argument(
        "--reset_num_timesteps",
        action="store_true",
        default=False,
        help="Reset internal timestep counter when resuming from --ckpt_path.",
    )
    parser.add_argument("--stop_wandb", action="store_true", default=False)
    parser.add_argument("--disable_progress_vis", action="store_true", default=False)
    parser.add_argument("--progress_vis_freq", type=int, default=50_000)
    parser.add_argument("--progress_vis_env_id", type=int, default=0)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="WANDB_MODE override. Defaults to offline.",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = args.sim_device or args.rl_device or "cuda:0"
    return args


def main():
    args = parse_args()
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    use_wandb = not args.stop_wandb
    active_team_name = os.environ.get("WANDB_ENTITY", default_team_name)
    active_project_name = os.environ.get("WANDB_PROJECT", default_project_name)

    exp_name = "train_gleam_stage1_lab"
    seed = int(args.seed)
    trial_name = f"{exp_name}_{get_time_str()}" if len(args.exp_name) == 0 else f"{args.exp_name}_{get_time_str()}"
    log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs_lab", trial_name)
    print(f"[LOGGING] Start logging into {log_dir}")

    env_cfg = build_stage1_lab_config(args)
    env_cfg_dict = dict(vars(env_cfg))
    env = EnvWrapperGLEAMLab(GLEAMStage1MockEnv(env_cfg))
    ckpt_path = None
    if args.ckpt_path:
        ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt_path))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--ckpt_path does not exist: {ckpt_path}")

    config = dict(
        algo=dict(
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
            target_kl=args.target_kl if args.use_target_kl else None,
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
            verbose=2,
            seed=seed,
            device=args.device,
        ),
        backend=args.backend,
        project_name=active_project_name,
        team_name=active_team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
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
    if not args.disable_progress_vis:
        callbacks.append(
            ProgressVisCallback(
                save_path=os.path.join(log_dir, "progress"),
                save_freq=args.progress_vis_freq,
                env_id=args.progress_vis_env_id,
                verbose=1,
            )
        )
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=exp_name,
                project_name=active_project_name,
                config={**config, **env_cfg_dict},
            )
        )
    callbacks = CallbackList(callbacks)

    if ckpt_path:
        print(f"[RESUME] Loading checkpoint from {ckpt_path}")
        model = PPO_Grid_Obs(**config["algo"])
        model.set_parameters(ckpt_path, exact_match=False, device=args.device)
        data, _, _ = load_from_zip_file(ckpt_path, device=args.device)
        for key in ("num_timesteps", "_episode_num", "_n_updates", "_current_progress_remaining"):
            if key in data:
                setattr(model, key, data[key])
        model.tensorboard_log = log_dir
        model.target_kl = args.target_kl if args.use_target_kl else None
    else:
        model = PPO_Grid_Obs(**config["algo"])

    timesteps_per_iter = args.num_envs * model.n_steps * 2
    additional_timesteps = timesteps_per_iter * args.total_iters
    reset_num_timesteps = True if not ckpt_path else args.reset_num_timesteps
    if ckpt_path and not reset_num_timesteps:
        print(
            f"[RESUME] Continue training from {model.num_timesteps} steps "
            f"for +{additional_timesteps} steps."
        )
    else:
        print(f"[TRAIN] Training for {additional_timesteps} steps.")
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            eval_env=None,
            tb_log_name=exp_name,
            log_interval=1,
        )
    finally:
        env.close()


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()
    main()
    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"Total wall-clock time: {(t_end - t_start) / 60:.3f}min")
