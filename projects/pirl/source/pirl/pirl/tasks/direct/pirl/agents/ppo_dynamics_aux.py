import copy
import itertools
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.spaces.torch import compute_space_size


PPODynamicsAux_default_config = {
    # Enable/disable auxiliary dynamics loss
    "dynamics_loss_scale": 0.02,
    # Learning rate for dynamics head (if None, uses PPO learning_rate)
    "dynamics_learning_rate": None,
    # Hidden layers for dynamics head MLP
    "dynamics_hidden_layers": [128, 128],
    # Predict only the first N vec components as delta (e.g. [dot, cross, vx, vy, wz] => 5)
    "dynamics_target_dims": 5,
}


class PPODynamicsAux(PPO):
    """PPO with auxiliary dynamics prediction head.

    The auxiliary task predicts delta of compact vec features:
    (vec_{t+1}[:N] - vec_t[:N]) from (vec_t, action_t).
    """

    def __init__(
        self,
        models: Mapping[str, Any],
        memory: Optional[Any] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(PPODynamicsAux_default_config)

        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # Aux dynamics config
        self._dynamics_loss_scale = float(self.cfg.get("dynamics_loss_scale", 0.0))
        self._dynamics_learning_rate = self.cfg.get("dynamics_learning_rate", None)
        self._dynamics_hidden_layers = list(self.cfg.get("dynamics_hidden_layers", [128, 128]))
        self._dynamics_target_dims = int(self.cfg.get("dynamics_target_dims", 5))

        # Locate vec slice in flattened observation (dict keys are flattened in sorted order).
        self._vec_start, self._vec_size = self._infer_vec_slice(self.observation_space)
        self._vec_end = self._vec_start + self._vec_size
        self._dynamics_target_dims = min(self._dynamics_target_dims, self._vec_size)

        # Dynamics head: [vec_t, action_t] -> delta vec_t (first N dims)
        self.dynamics_model = None
        if self._dynamics_loss_scale > 0.0:
            num_actions = compute_space_size(self.action_space, occupied_size=True)
            in_dim = self._vec_size + num_actions
            layers = []
            prev = in_dim
            for h in self._dynamics_hidden_layers:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ELU())
                prev = h
            layers.append(nn.Linear(prev, self._dynamics_target_dims))
            self.dynamics_model = nn.Sequential(*layers).to(self.device)

            lr = self._learning_rate if self._dynamics_learning_rate is None else float(self._dynamics_learning_rate)
            self.optimizer.add_param_group({"params": self.dynamics_model.parameters(), "lr": lr})
            self.checkpoint_modules["dynamics_model"] = self.dynamics_model

    @staticmethod
    def _infer_vec_slice(observation_space) -> tuple[int, int]:
        if not isinstance(observation_space, gymnasium.spaces.Dict):
            raise ValueError("PPODynamicsAux expects Dict observation space with key 'vec'.")
        start = 0
        vec_start = None
        vec_size = None
        for key in sorted(observation_space.spaces.keys()):
            size = compute_space_size(observation_space.spaces[key], occupied_size=True)
            if key == "vec":
                vec_start = start
                vec_size = size
            start += size
        if vec_start is None or vec_size is None:
            raise ValueError("Dict observation space must contain key 'vec'.")
        return vec_start, vec_size

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        # Ensure next_states is stored in memory for dynamics supervision.
        if self.memory is not None:
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)

    def _update(self, timestep: int, timesteps: int) -> None:
        # Mostly identical to skrl PPO update, with + dynamics loss term.
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # returns/advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(
            names=["states", "actions", "log_prob", "values", "returns", "advantages", "next_states"],
            mini_batches=self._mini_batches,
        )

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_dynamics_loss = 0.0
        cumulative_dynamics_to_policy_grad_norm = 0.0

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            for (
                sampled_states_raw,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_next_states_raw,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states_raw, train=not epoch)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
                    # Keep a differentiable path from aux dynamics loss to policy/backbone.
                    # We use the policy mean action as a gradient carrier while keeping forward
                    # target alignment with executed action from rollout memory.
                    policy_mean_action = None
                    if hasattr(self.policy, "distribution"):
                        try:
                            policy_dist = self.policy.distribution(role="policy")
                            policy_mean_action = getattr(policy_dist, "mean", None)
                        except Exception:
                            policy_mean_action = None

                    with torch.no_grad():
                        ratio_kl = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio_kl) - 1) - ratio_kl).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0.0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    dynamics_loss = torch.tensor(0.0, device=self.device)
                    if self.dynamics_model is not None and self._dynamics_loss_scale > 0.0:
                        vec_t = sampled_states_raw[:, self._vec_start : self._vec_end]
                        vec_tp1 = sampled_next_states_raw[:, self._vec_start : self._vec_end]
                        dyn_actions = sampled_actions
                        if (
                            policy_mean_action is not None
                            and isinstance(policy_mean_action, torch.Tensor)
                            and policy_mean_action.shape == sampled_actions.shape
                        ):
                            # Forward value remains sampled_actions, gradients flow through policy mean.
                            dyn_actions = sampled_actions + (policy_mean_action - policy_mean_action.detach())
                        dyn_in = torch.cat([vec_t, dyn_actions], dim=-1)
                        delta_true = vec_tp1[:, : self._dynamics_target_dims] - vec_t[:, : self._dynamics_target_dims]
                        delta_pred = self.dynamics_model(dyn_in)
                        dynamics_loss = self._dynamics_loss_scale * F.mse_loss(delta_pred, delta_true)

                    # Diagnostic metric: how strongly aux loss pushes policy/backbone params.
                    dynamics_to_policy_grad_norm = 0.0
                    if dynamics_loss.requires_grad:
                        policy_params = tuple(self.policy.parameters())
                        aux_grads = torch.autograd.grad(
                            dynamics_loss,
                            policy_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        grad_sq_sum = torch.tensor(0.0, device=self.device)
                        for grad in aux_grads:
                            if grad is not None:
                                grad_sq_sum = grad_sq_sum + grad.pow(2).sum()
                        dynamics_to_policy_grad_norm = float(torch.sqrt(grad_sq_sum).detach().item())

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + dynamics_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        params = self.policy.parameters()
                    else:
                        params = itertools.chain(self.policy.parameters(), self.value.parameters())
                        if self.dynamics_model is not None:
                            params = itertools.chain(params, self.dynamics_model.parameters())
                    nn.utils.clip_grad_norm_(params, self._grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += float(policy_loss.item())
                cumulative_value_loss += float(value_loss.item())
                cumulative_dynamics_loss += float(dynamics_loss.item())
                cumulative_dynamics_to_policy_grad_norm += dynamics_to_policy_grad_norm
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(entropy_loss.item())

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        denom = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)
        if self.dynamics_model is not None and self._dynamics_loss_scale > 0.0:
            self.track_data("Loss / Dynamics loss", cumulative_dynamics_loss / denom)
            self.track_data("Grad / Dynamics-to-policy norm", cumulative_dynamics_to_policy_grad_norm / denom)

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
