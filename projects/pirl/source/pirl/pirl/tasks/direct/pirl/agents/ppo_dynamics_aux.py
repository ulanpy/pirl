"""PPO-RNN agent with auxiliary dynamics head and optional HJB regularizer on the critic."""

import copy
import itertools
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.spaces.torch import compute_space_size

from .obs_layout import get_vec_costmap_layout


class _MinibatchLosses(NamedTuple):
    """Per-minibatch PPO, auxiliary, and diagnostic scalars for one backward step."""

    policy_loss: torch.Tensor
    entropy_loss: torch.Tensor
    value_loss: torch.Tensor
    dynamics_loss: torch.Tensor
    hjb_loss: torch.Tensor
    kl_divergence: torch.Tensor
    dynamics_to_policy_grad_norm: float


# Merged into skrl PPO_DEFAULT_CONFIG; overridden by YAML ``agent:`` section.
PPODynamicsAux_default_config = {
    # Enable/disable auxiliary dynamics loss
    "dynamics_loss_scale": 0.02,
    # Learning rate for dynamics head (if None, uses PPO learning_rate)
    "dynamics_learning_rate": None,
    # Hidden layers for dynamics head MLP
    "dynamics_hidden_layers": [128, 128],
    # Predict only the first N vec components as delta (e.g. [dot, cross, vx, vy, wz] => 5)
    "dynamics_target_dims": 5,
    # Use normalized states (state_preprocessor) for dynamics supervision.
    # This keeps aux targets/input scales stable during long training.
    "dynamics_use_normalized_vec": True,
    # HJB regularizer (PINN-style) on value function.
    # If 0.0, HJB branch is disabled.
    "hjb_loss_scale": 0.5,
    # Differential-drive kinematics scales for normalized actions.
    "hjb_max_lin_vel": 0.5,
    "hjb_max_ang_vel": 3.0,
    # Running cost weights in Hamiltonian.
    "hjb_time_weight": 1.0,
    "hjb_distance_weight": 0.2,
    "hjb_heading_weight": 0.2,
    "hjb_control_weight": 0.05,
    # Progress-consistent term in running cost (larger -> stronger forward preference).
    "hjb_progress_weight": 1.0,
    # Hamiltonian mode:
    # - "policy": evaluate H(x, u_policy, grad V)
    # - "optimal": evaluate H*(x, grad V) using analytic argmin dH/du = 0
    "hjb_hamiltonian_mode": "optimal",
    # Preferred indices in vec for HJB state [d, psi].
    # Current vec layout:
    # [dot, cross, vx, vy, wz, d_signed, heading_error, path_obs..., prev_action, prev_reward]
    "hjb_vec_d_index": 5,
    "hjb_vec_psi_index": 6,
}


class PPODynamicsAuxRNN(PPO_RNN):
    """PPO-RNN (skrl) extended with optional dynamics MLP and HJB-style critic penalty.

    Dynamics auxiliary: predicts ``vec_{t+1}[:N] - vec_t[:N]`` from ``(vec_t, a_t)`` with a
    gradient path through the policy mean when enabled. HJB branch (if ``hjb_loss_scale > 0``)
    penalizes a local Hamiltonian residual built from selected ``vec`` indices and analytic
    relative kinematics; see ``HJB_THEORY_TIME_DISTANCE.md`` in this package.
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
        """Merge defaults with ``PPODynamicsAux_default_config`` and ``cfg``, build ``dynamics_model`` if needed."""
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG) # deepcopy потому что словари мутабельные 
        _cfg.update(PPODynamicsAux_default_config) # дефолтные конфиги для PPO Dynamics Aux

        _cfg.update(cfg if cfg is not None else {}) # обновляем дефолтные конфиги пользовательскими, если они есть
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        ) # вызываем конструктор базового класса PPO с обновленным конфигом

        # Aux dynamics config
        self._dynamics_loss_scale = float(self.cfg.get("dynamics_loss_scale", 0.0))
        self._dynamics_learning_rate = self.cfg.get("dynamics_learning_rate", None)
        self._dynamics_hidden_layers = list(self.cfg.get("dynamics_hidden_layers", [128, 128]))
        self._dynamics_target_dims = int(self.cfg.get("dynamics_target_dims", 5))
        self._dynamics_use_normalized_vec = bool(self.cfg.get("dynamics_use_normalized_vec", True))
        self._hjb_loss_scale = float(self.cfg.get("hjb_loss_scale", 0.0))
        self._hjb_max_lin_vel = float(self.cfg.get("hjb_max_lin_vel", 0.5))
        self._hjb_max_ang_vel = float(self.cfg.get("hjb_max_ang_vel", 3.0))
        self._hjb_time_weight = float(self.cfg.get("hjb_time_weight", 1.0))
        self._hjb_distance_weight = float(self.cfg.get("hjb_distance_weight", 0.2))
        self._hjb_heading_weight = float(self.cfg.get("hjb_heading_weight", 0.2))
        self._hjb_control_weight = float(self.cfg.get("hjb_control_weight", 0.05))
        self._hjb_progress_weight = float(self.cfg.get("hjb_progress_weight", 1.0))
        self._hjb_hamiltonian_mode = str(self.cfg.get("hjb_hamiltonian_mode", "optimal")).strip().lower()
        if self._hjb_hamiltonian_mode not in ("policy", "optimal"):
            raise ValueError(
                "hjb_hamiltonian_mode must be one of {'policy', 'optimal'}, "
                f"got: {self._hjb_hamiltonian_mode}"
            )
        self._hjb_vec_d_index = int(self.cfg.get("hjb_vec_d_index", 5))
        self._hjb_vec_psi_index = int(self.cfg.get("hjb_vec_psi_index", 6))

        # Vec slice in flattened Dict(obs); same layout as recurrent_models / skrl preprocessor.
        self._vec_start, self._vec_size, _, _ = get_vec_costmap_layout(self.observation_space)
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

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Register ``next_states`` in rollout memory for dynamics supervision."""
        super().init(trainer_cfg=trainer_cfg)
        # Ensure next_states is stored in memory for dynamics supervision.
        if self.memory is not None:
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            if "next_states" not in self._tensors_names:
                self._tensors_names.append("next_states")

    @staticmethod
    def _compute_gae(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        bootstrap_values: torch.Tensor,
        discount_factor: float,
        lambda_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TD(λ) advantages and returns over the rollout buffer; last step bootstrapped with ``bootstrap_values``."""
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]
        for i in reversed(range(memory_size)):
            next_v = values[i + 1] if i < memory_size - 1 else bootstrap_values
            advantage = (
                rewards[i]
                - values[i]
                + discount_factor * not_dones[i] * (next_v + lambda_coefficient * advantage)
            )
            advantages[i] = advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _bootstrap_last_values(self) -> torch.Tensor:
        """Evaluate critic on stored ``_current_next_states`` (inverse value preprocessor on output)."""
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value"
            )
            self.value.train(True)
            return self._value_preprocessor(last_values, inverse=True)

    def _rnn_kwargs_for_minibatch(
        self,
        batch_index: int,
        sampled_rnn_batches: List[List[torch.Tensor]],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """Build ``rnn`` / ``terminated`` kwargs for policy and value for one training minibatch."""
        if not self._rnn:
            return {}, {}
        done = terminated | truncated
        if self.policy is self.value:
            rnn_policy = {
                "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[batch_index]],
                "terminated": done,
            }
            return rnn_policy, rnn_policy
        rnn_policy = {
            "rnn": [
                s.transpose(0, 1)
                for s, n in zip(sampled_rnn_batches[batch_index], self._rnn_tensors_names)
                if "policy" in n
            ],
            "terminated": done,
        }
        rnn_value = {
            "rnn": [
                s.transpose(0, 1)
                for s, n in zip(sampled_rnn_batches[batch_index], self._rnn_tensors_names)
                if "value" in n
            ],
            "terminated": done,
        }
        return rnn_policy, rnn_value

    def _policy_mean_for_dynamics_grad(self) -> Optional[torch.Tensor]:
        """Gaussian policy mean after ``act``; used to route dynamics loss gradients into the policy."""
        if not hasattr(self.policy, "distribution"):
            return None
        try:
            policy_dist = self.policy.distribution(role="policy")
            return getattr(policy_dist, "mean", None)
        except Exception:
            return None

    def _dynamics_loss(
        self,
        sampled_states: torch.Tensor,
        sampled_states_raw: torch.Tensor,
        sampled_next_states_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        policy_mean_action: Optional[torch.Tensor],
        train_preprocessor: bool,
    ) -> torch.Tensor:
        """Scaled MSE between predicted and true ``vec`` deltas (first ``dynamics_target_dims``)."""
        if self.dynamics_model is None or self._dynamics_loss_scale <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if self._dynamics_use_normalized_vec:
            sampled_next_states = self._state_preprocessor(sampled_next_states_raw, train=train_preprocessor)
            vec_t = sampled_states[:, self._vec_start : self._vec_end]
            vec_tp1 = sampled_next_states[:, self._vec_start : self._vec_end]
        else:
            vec_t = sampled_states_raw[:, self._vec_start : self._vec_end]
            vec_tp1 = sampled_next_states_raw[:, self._vec_start : self._vec_end]
        dyn_actions = sampled_actions
        if (
            policy_mean_action is not None
            and isinstance(policy_mean_action, torch.Tensor)
            and policy_mean_action.shape == sampled_actions.shape
        ):
            dyn_actions = sampled_actions + (policy_mean_action - policy_mean_action.detach())
        dyn_in = torch.cat([vec_t, dyn_actions], dim=-1)
        delta_true = vec_tp1[:, : self._dynamics_target_dims] - vec_t[:, : self._dynamics_target_dims]
        delta_pred = self.dynamics_model(dyn_in)
        return self._dynamics_loss_scale * F.mse_loss(delta_pred, delta_true)

    def _hjb_loss(
        self,
        sampled_states_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        sampled_terminated: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled mean squared Hamiltonian residual w.r.t. raw ``vec`` slice (FP32); zero if disabled."""
        if self._hjb_loss_scale <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if self._hjb_vec_d_index < 0 or self._hjb_vec_psi_index < 0:
            raise ValueError("hjb_vec_d_index and hjb_vec_psi_index must be non-negative.")
        with torch.autocast(device_type=self._device_type, enabled=False):
            hjb_states_raw = sampled_states_raw.detach().clone().float().requires_grad_(True)
            hjb_states = self._state_preprocessor(hjb_states_raw, train=False)
            hjb_rnn_value: dict = {}
            if self._rnn:
                hjb_rnn_value = {
                    "rnn": self._rnn_initial_states["value"],
                    "terminated": torch.zeros_like(sampled_terminated, dtype=torch.bool),
                }
            hjb_values, _, _ = self.value.act({"states": hjb_states, **hjb_rnn_value}, role="value")
            hjb_vec_raw = hjb_states_raw[:, self._vec_start : self._vec_end]
            grad_vec_full = torch.autograd.grad(
                hjb_values.sum(),
                hjb_vec_raw,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_vec = torch.zeros_like(hjb_vec_raw) if grad_vec_full is None else grad_vec_full
            vec = hjb_vec_raw
            if vec.shape[1] <= max(self._hjb_vec_d_index, self._hjb_vec_psi_index):
                raise ValueError(
                    "HJB vec indices out of range for current vec layout: "
                    f"vec_dim={vec.shape[1]}, d_idx={self._hjb_vec_d_index}, psi_idx={self._hjb_vec_psi_index}"
                )
            # HJB state: x = [d, psi], where d is signed cross-track error and psi is heading error.
            d_err = vec[:, self._hjb_vec_d_index : self._hjb_vec_d_index + 1]
            psi_err = vec[:, self._hjb_vec_psi_index : self._hjb_vec_psi_index + 1]
            dVdd = grad_vec[:, self._hjb_vec_d_index : self._hjb_vec_d_index + 1]
            dVdpsi = grad_vec[:, self._hjb_vec_psi_index : self._hjb_vec_psi_index + 1]
            control_w_t = torch.tensor(
                float(self._hjb_control_weight),
                device=self.device,
                dtype=torch.float32,
            ).clamp(min=1e-6)
            if self._hjb_hamiltonian_mode == "optimal":
                # Hamiltonian minimization (no CBF): solve dH/du = 0 for unconstrained u* = [v*, w*].
                # H(v, w) = l(d, psi, v, w) + dV/dd * d_dot + dV/dpsi * psi_dot
                # with d_dot = v*sin(psi), psi_dot = w and
                # l = w_t + w_d*d^2 + w_psi*(1-cos(psi)) + w_u*(v^2 + 0.1*w^2) - w_p*v*cos(psi)
                v_ctrl = (
                    self._hjb_progress_weight * torch.cos(psi_err) - dVdd * torch.sin(psi_err)
                ) / (2.0 * control_w_t)
                w_ctrl = -dVdpsi / (0.2 * control_w_t)
            else:
                # Policy-evaluated Hamiltonian (on-policy residual).
                v_ctrl = sampled_actions[:, 0:1].float() * self._hjb_max_lin_vel
                w_ctrl = sampled_actions[:, 1:2].float() * self._hjb_max_ang_vel
            # Signed-error kinematics (small-curvature approximation in local path frame):
            # d_dot = v * sin(psi), psi_dot = w.
            d_dot = v_ctrl * torch.sin(psi_err)
            psi_dot = w_ctrl
            # Running cost consistent with reward priorities:
            # penalize cross-track and heading error, control effort,
            # and reward forward progress along heading target via -v*cos(psi).
            control_cost = v_ctrl * v_ctrl + 0.1 * (w_ctrl * w_ctrl)
            running_cost = (
                self._hjb_time_weight
                + self._hjb_distance_weight * (d_err * d_err)
                + self._hjb_heading_weight * (1.0 - torch.cos(psi_err))
                + control_w_t * control_cost
                - self._hjb_progress_weight * (v_ctrl * torch.cos(psi_err))
            )
            hamiltonian = running_cost + dVdd * d_dot + dVdpsi * psi_dot
            return self._hjb_loss_scale * torch.mean(hamiltonian * hamiltonian)

    def _dynamics_to_policy_grad_norm(self, dynamics_loss: torch.Tensor) -> float:
        """L2 norm of ``dynamics_loss`` gradients w.r.t. policy parameters (logging only)."""
        if not dynamics_loss.requires_grad:
            return 0.0
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
        return float(torch.sqrt(grad_sq_sum).detach().item())

    def _minibatch_losses(
        self,
        epoch: int,
        sampled_states_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        sampled_terminated: torch.Tensor,
        sampled_truncated: torch.Tensor,
        sampled_log_prob: torch.Tensor,
        sampled_values: torch.Tensor,
        sampled_returns: torch.Tensor,
        sampled_advantages: torch.Tensor,
        sampled_next_states_raw: torch.Tensor,
        rnn_policy: dict,
        rnn_value: dict,
    ) -> _MinibatchLosses:
        """Forward PPO policy/value, dynamics, HJB; return losses and KL for one minibatch."""
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            sampled_states = self._state_preprocessor(sampled_states_raw, train=not epoch)
            _, next_log_prob, _ = self.policy.act(
                {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
            )
            policy_mean_action = self._policy_mean_for_dynamics_grad()

            with torch.no_grad():
                ratio_kl = next_log_prob - sampled_log_prob
                kl_divergence = ((torch.exp(ratio_kl) - 1) - ratio_kl).mean()

            if self._entropy_loss_scale:
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
            else:
                entropy_loss = torch.tensor(0.0, device=self.device)

            ratio = torch.exp(next_log_prob - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * torch.clip(
                ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
            )
            policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

            predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")
            if self._clip_predicted_values:
                predicted_values = sampled_values + torch.clip(
                    predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                )
            value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

            dynamics_loss = self._dynamics_loss(
                sampled_states,
                sampled_states_raw,
                sampled_next_states_raw,
                sampled_actions,
                policy_mean_action,
                train_preprocessor=not epoch,
            )
            hjb_loss = self._hjb_loss(sampled_states_raw, sampled_actions, sampled_terminated)
            dyn_grad_norm = self._dynamics_to_policy_grad_norm(dynamics_loss)

        return _MinibatchLosses(
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            value_loss=value_loss,
            dynamics_loss=dynamics_loss,
            hjb_loss=hjb_loss,
            kl_divergence=kl_divergence,
            dynamics_to_policy_grad_norm=dyn_grad_norm,
        )

    def _optimizer_step(self, total_loss: torch.Tensor) -> None:
        """``backward`` on combined losses, distributed grad sync, optional clip, ``optimizer.step``."""
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

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

    def _schedule_learning_rate(self, kl_divergences: List[torch.Tensor]) -> None:
        """Step LR scheduler once per epoch (KL-adaptive uses mean KL over minibatches)."""
        if not self._learning_rate_scheduler:
            return
        if isinstance(self.scheduler, KLAdaptiveLR):
            kl = torch.tensor(kl_divergences, device=self.device).mean()
            if config.torch.is_distributed:
                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                kl /= config.torch.world_size
            self.scheduler.step(kl.item())
        else:
            self.scheduler.step()

    def _track_update_metrics(
        self,
        cumulative_policy_loss: float,
        cumulative_value_loss: float,
        cumulative_entropy_loss: float,
        cumulative_dynamics_loss: float,
        cumulative_hjb_loss: float,
        cumulative_dynamics_to_policy_grad_norm: float,
        denom: float,
    ) -> None:
        """Write averaged losses and policy stats to experiment tracking (e.g. TensorBoard)."""
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)
        if self.dynamics_model is not None and self._dynamics_loss_scale > 0.0:
            self.track_data("Loss / Dynamics loss", cumulative_dynamics_loss / denom)
            self.track_data("Grad / Dynamics-to-policy norm", cumulative_dynamics_to_policy_grad_norm / denom)
        if self._hjb_loss_scale > 0.0:
            self.track_data("Loss / HJB loss", cumulative_hjb_loss / denom)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def _update(self, timestep: int, timesteps: int) -> None:
        """One PPO-RNN update: GAE, sequence minibatches, combined loss and optimizer steps.

        ``timestep`` / ``timesteps`` are unused but kept for API compatibility with skrl ``Agent``.
        """
        last_values = self._bootstrap_last_values()
        values = self.memory.get_tensor_by_name("values")
        returns, advantages = self._compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            bootstrap_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(
            names=[
                "states",
                "actions",
                "terminated",
                "truncated",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "next_states",
            ],
            mini_batches=self._mini_batches,
            sequence_length=self._rnn_sequence_length,
        )
        sampled_rnn_batches: List[List[torch.Tensor]] = []
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_dynamics_loss = 0.0
        cumulative_hjb_loss = 0.0
        cumulative_dynamics_to_policy_grad_norm = 0.0

        for epoch in range(self._learning_epochs):
            kl_divergences: List[torch.Tensor] = []
            for i, (
                sampled_states_raw,
                sampled_actions,
                sampled_terminated,
                sampled_truncated,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_next_states_raw,
            ) in enumerate(sampled_batches):
                rnn_policy, rnn_value = self._rnn_kwargs_for_minibatch(
                    i, sampled_rnn_batches, sampled_terminated, sampled_truncated
                )
                L = self._minibatch_losses(
                    epoch,
                    sampled_states_raw,
                    sampled_actions,
                    sampled_terminated,
                    sampled_truncated,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                    sampled_next_states_raw,
                    rnn_policy,
                    rnn_value,
                )
                kl_divergences.append(L.kl_divergence)

                if self._kl_threshold and L.kl_divergence > self._kl_threshold:
                    break

                total = L.policy_loss + L.entropy_loss + L.value_loss + L.dynamics_loss + L.hjb_loss
                self._optimizer_step(total)

                cumulative_policy_loss += float(L.policy_loss.item())
                cumulative_value_loss += float(L.value_loss.item())
                cumulative_dynamics_loss += float(L.dynamics_loss.item())
                cumulative_hjb_loss += float(L.hjb_loss.item())
                cumulative_dynamics_to_policy_grad_norm += L.dynamics_to_policy_grad_norm
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(L.entropy_loss.item())

            self._schedule_learning_rate(kl_divergences)

        denom = float(self._learning_epochs * self._mini_batches)
        self._track_update_metrics(
            cumulative_policy_loss,
            cumulative_value_loss,
            cumulative_entropy_loss,
            cumulative_dynamics_loss,
            cumulative_hjb_loss,
            cumulative_dynamics_to_policy_grad_norm,
            denom,
        )


# Backward-compatible alias for config/entry-points that still use old class name.
PPODynamicsAux = PPODynamicsAuxRNN
