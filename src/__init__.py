import gymnasium as gym
from . import envs, tasks

# Регистрация задач для Isaac Lab
# Это позволит запускать обучение командой:
# python train.py --task Isaac-Velocity-Caterpillar-v0

gym.register(
    id="Isaac-Velocity-Caterpillar-v0",
    entry_point="pinn_nav.envs:CaterpillarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "pinn_nav.tasks:CaterpillarEnvCfg",
        "skrl_cfg_entry_point": "pinn_nav.tasks:skrl_sac_cfg.yaml",
    },
)
