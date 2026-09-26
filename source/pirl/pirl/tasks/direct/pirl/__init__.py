# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .manager_based import PirlManagerEnv, PirlManagerEnvCfg

##
# Register Gym environments.
##


gym.register(
    id="burger",
    entry_point=PirlManagerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PirlManagerEnvCfg,
        "skrl_ppo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_ppo_rnn_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_rnn_cfg.yaml",
    },
)
