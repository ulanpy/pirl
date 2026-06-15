import copy
from typing import Any, Mapping, Union, cast

from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.utils.runner.torch import Runner


def _migrate_agent_cfg(agent_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply skrl 2.x YAML field renames for custom agent construction."""
    agent_cfg = copy.deepcopy(agent_cfg)
    if "lambda" in agent_cfg:
        agent_cfg.setdefault("gae_lambda", agent_cfg.pop("lambda"))
    if "clip_predicted_values" in agent_cfg:
        agent_cfg.pop("clip_predicted_values")
    if "rewards_shaper_scale" in agent_cfg:
        agent_cfg.pop("rewards_shaper_scale")
    if "observation_preprocessor" not in agent_cfg and "state_preprocessor" in agent_cfg:
        agent_cfg["observation_preprocessor"] = agent_cfg.pop("state_preprocessor")
        if "state_preprocessor_kwargs" in agent_cfg:
            agent_cfg["observation_preprocessor_kwargs"] = agent_cfg.pop("state_preprocessor_kwargs")
    return agent_cfg


def get_runner(env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], ml_framework: str) -> Runner:
    """Universal runner factory for any agent class."""
    if ml_framework.startswith("torch"):
        class UniversalRunner(Runner):
            def _component(self, name: str):
                lname = name.lower()

                from .ppo_hjb_rnn import PPOHjbRNN, PPOHjbRNN_default_config
                from .recurrent_models import (
                    FeedForwardDeterministicValue,
                    RecurrentGaussianPolicy,
                )
                custom_components = {
                    "ppohjbrnn": PPOHjbRNN,
                    "ppohjbrnn_default_config": PPOHjbRNN_default_config,
                    "recurrentgaussianpolicy": RecurrentGaussianPolicy,
                    "feedforwarddeterministicvalue": FeedForwardDeterministicValue,
                }
                if lname in custom_components:
                    return custom_components[lname]
                return super()._component(name)

            def _generate_models(self, env, cfg):
                models_cfg = cfg.get("models", {})
                for role, role_cfg in models_cfg.items():
                    if not isinstance(role_cfg, dict):
                        continue
                    model_class = str(role_cfg.get("class", "")).lower()
                    if model_class in [
                        "recurrentgaussianpolicy",
                        "recurrentdeterministicvalue",
                    ]:
                        role_cfg.setdefault("num_envs", env.num_envs)
                models = cast(dict[str, dict[str, Any]], super()._generate_models(env, cfg))
                return models

            def _generate_agent(self, env, cfg, models):
                cfg = cast(dict[str, Any], copy.deepcopy(cfg))
                agent_class_name = cfg.get("agent", {}).get("class", "")

                standard_agents = ["a2c", "amp", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]
                if agent_class_name.lower() in standard_agents:
                    return super()._generate_agent(env, cfg, models)

                device = env.device
                agent_id = "agent"

                if "memory" not in cfg:
                    cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
                memory_cfg = copy.deepcopy(cfg["memory"])
                memory_class = self._component(memory_cfg.pop("class", "RandomMemory"))
                if memory_cfg["memory_size"] < 0:
                    memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
                memory = memory_class(num_envs=env.num_envs, device=device, **self._process_cfg(memory_cfg))

                base_cfg = self._component(f"{agent_class_name.lower()}_default_config")
                agent_cfg = _migrate_agent_cfg(base_cfg.copy())
                agent_cfg.update(_migrate_agent_cfg(self._process_cfg(cfg["agent"])))

                obs_space = env.observation_space
                state_space = getattr(env, "state_space", None)
                agent_cfg.setdefault("observation_preprocessor_kwargs", {})
                agent_cfg["observation_preprocessor_kwargs"].update({"size": obs_space, "device": device})
                agent_cfg.setdefault("value_preprocessor_kwargs", {})
                agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
                if state_space is not None:
                    agent_cfg.setdefault("state_preprocessor_kwargs", {})
                    agent_cfg["state_preprocessor_kwargs"].update({"size": state_space, "device": device})

                agent_class = self._component(agent_class_name)
                return agent_class(
                    models=models[agent_id],
                    memory=memory,
                    observation_space=obs_space,
                    state_space=state_space,
                    action_space=env.action_space,
                    cfg=agent_cfg,
                    device=device,
                )

        return UniversalRunner(env, cfg)

    return Runner(env, cfg)
