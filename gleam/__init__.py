"""GLEAM package.

Legacy Isaac Gym task registration is best-effort so lightweight modules
(`gleam.callback`, `gleam.network.*`) can still be imported on new stacks.
"""

task_registry = None
_GLEAM_REGISTRATION_ERROR = None

try:
    from legged_gym.utils.task_registry import task_registry
    from gleam.env.config_gleam import Config_GLEAM, DroneCfgPPO
    from gleam.env.config_gleam_eval import Config_GLEAM_Eval
    from gleam.env.env_gleam_eval import Env_GLEAM_Eval
    from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1
    from gleam.env.env_gleam_stage2 import Env_GLEAM_Stage2

    task_registry.register("train_gleam_stage1", Env_GLEAM_Stage1, Config_GLEAM, DroneCfgPPO)
    task_registry.register("train_gleam_stage2", Env_GLEAM_Stage2, Config_GLEAM, DroneCfgPPO)
    task_registry.register("eval_gleam_gleambench", Env_GLEAM_Eval, Config_GLEAM_Eval, DroneCfgPPO)
except Exception as exc:  # pragma: no cover - depends on legacy stack availability
    _GLEAM_REGISTRATION_ERROR = exc
