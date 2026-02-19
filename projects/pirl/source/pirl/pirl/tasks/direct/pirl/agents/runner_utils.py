import copy
from typing import Any, Mapping, Union

from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper

def get_runner(env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], ml_framework: str) -> Runner:
    """Universal runner factory for any agent class."""
    if ml_framework.startswith("torch"):
        class UniversalRunner(Runner):
            def _component(self, name: str):
                lname = name.lower()

                # Explicit custom component registry (hardcoded on purpose).
                from .ppo_dynamics_aux import PPODynamicsAux, PPODynamicsAux_default_config
                custom_components = {
                    "ppodynamicsaux": PPODynamicsAux,
                    "ppodynamicsaux_default_config": PPODynamicsAux_default_config,
                }
                if lname in custom_components:
                    return custom_components[lname]
                return super()._component(name)

            def _generate_agent(self, env, cfg, models):
                agent_class_name = cfg.get("agent", {}).get("class", "")
                
                # Check if it's a standard skrl agent
                standard_agents = ["a2c", "amp", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]
                if agent_class_name.lower() in standard_agents:
                    return super()._generate_agent(env, cfg, models)

                # --- Generic initialization for custom agents ---
                device = env.device
                agent_id = "agent"

                if "memory" not in cfg:
                    cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
                memory_cfg = copy.deepcopy(cfg["memory"])
                memory_class = self._component(memory_cfg.pop("class", "RandomMemory"))
                if memory_cfg["memory_size"] < 0:
                    memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
                memory = memory_class(num_envs=env.num_envs, device=device, **self._process_cfg(memory_cfg))

                # Build agent cfg from agent's explicit default config.
                base_cfg = self._component(f"{agent_class_name.lower()}_default_config")
                
                agent_cfg = base_cfg.copy()
                agent_cfg.update(self._process_cfg(cfg["agent"]))
                agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": env.observation_space, "device": device})
                agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})

                agent_class = self._component(agent_class_name)
                return agent_class(
                    models=models[agent_id],
                    memory=memory,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    cfg=agent_cfg,
                    device=device,
                )

        return UniversalRunner(env, cfg)
    
    return Runner(env, cfg)
