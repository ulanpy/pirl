"""PPO-RNN agent with optional HJB PINN-style regularizer on the critic (skrl 2.x)."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN, compute_gae
from skrl.resources.schedulers.torch import KLAdaptiveLR

from .obs_layout import get_vec_costmap_layout
from .ppo_hjb_cfg import PPOHjbRNN_CFG


class _MinibatchLosses(NamedTuple):
    """Per-minibatch PPO and HJB scalars for one backward step."""

    policy_loss: torch.Tensor
    entropy_loss: torch.Tensor
    value_loss: torch.Tensor
    hjb_loss: torch.Tensor
    kl_divergence: torch.Tensor
    hjb_residual_abs_mean: float
    hjb_value_abs_mean: float
    hjb_running_cost_mean: float
    hjb_lidar_grad_term_abs_mean: float


class _HjbResult(NamedTuple):
    """HJB loss and detached diagnostics for one minibatch."""

    loss: torch.Tensor
    residual_abs_mean: float
    value_abs_mean: float
    running_cost_mean: float
    lidar_grad_term_abs_mean: float


_PPOHJB_FIELD_NAMES = {f.name for f in dataclasses.fields(PPOHjbRNN_CFG)}


def _build_agent_cfg(raw: Mapping[str, Any] | None) -> PPOHjbRNN_CFG:
    """Merge YAML/dict overrides into ``PPOHjbRNN_CFG`` defaults."""
    merged = dataclasses.asdict(PPOHjbRNN_CFG())
    if raw:
        merged.update({k: v for k, v in raw.items() if k in _PPOHJB_FIELD_NAMES})
    cfg = PPOHjbRNN_CFG(**merged)
    cfg.expand()
    return cfg


# Runner default-config hook (``ppohjbrnn_default_config``).
PPOHjbRNN_default_config = {
    k: v
    for k, v in dataclasses.asdict(PPOHjbRNN_CFG()).items()
    if k not in {"experiment"}
}


class PPOHjbRNN(PPO_RNN):
    """PPO-RNN (skrl 2.x) extended with an optional HJB-style critic penalty."""

    def __init__(
        self,
        models: Mapping[str, Any],
        memory: Optional[Any] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: PPOHjbRNN_CFG | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(cfg, PPOHjbRNN_CFG):
            cfg_obj = copy.deepcopy(cfg)
            cfg_obj.expand()
        else:
            cfg_obj = _build_agent_cfg(cfg if isinstance(cfg, Mapping) else None)
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg_obj,
        )
        self.cfg: PPOHjbRNN_CFG
        self._init_hjb_fields()

    def _init_hjb_fields(self) -> None:
        """Load HJB hyperparameters from the extended agent config."""
        self._hjb_loss_scale = float(self.cfg.hjb_loss_scale)
        self._hjb_max_lin_vel = float(self.cfg.hjb_max_lin_vel)
        self._hjb_max_ang_vel = float(self.cfg.hjb_max_ang_vel)
        self._hjb_time_weight = float(self.cfg.hjb_time_weight)
        self._hjb_distance_weight = float(self.cfg.hjb_distance_weight)
        self._hjb_heading_weight = float(self.cfg.hjb_heading_weight)
        self._hjb_control_weight = float(self.cfg.hjb_control_weight)
        self._hjb_progress_weight = float(self.cfg.hjb_progress_weight)
        self._hjb_hamiltonian_mode = str(self.cfg.hjb_hamiltonian_mode).strip().lower()
        if self._hjb_hamiltonian_mode not in ("policy", "optimal"):
            raise ValueError(
                "hjb_hamiltonian_mode must be one of {'policy', 'optimal'}, "
                f"got: {self._hjb_hamiltonian_mode}"
            )
        self._hjb_vec_d_index = int(self.cfg.hjb_vec_d_index)
        self._hjb_vec_psi_index = int(self.cfg.hjb_vec_psi_index)
        self._hjb_lidar_start = int(self.cfg.hjb_lidar_hits_start_index)
        self._hjb_lidar_K = int(self.cfg.hjb_lidar_sector_count)
        if self._hjb_lidar_K > 0 and self._hjb_lidar_start < 0:
            raise ValueError(
                "hjb_lidar_sector_count > 0 requires a non-negative hjb_lidar_hits_start_index."
            )
        self._hjb_step_dt = float(self.cfg.hjb_step_dt)
        _gamma = float(self.cfg.discount_factor)
        self._hjb_discount_rate = (
            -math.log(max(min(_gamma, 1.0 - 1e-9), 1e-9)) / max(self._hjb_step_dt, 1e-9)
        )
        self._vec_start, self._vec_size, _, _ = get_vec_costmap_layout(self.observation_space)
        self._vec_end = self._vec_start + self._vec_size

    @staticmethod
    def _preprocessed_inputs(
        agent: PPOHjbRNN,
        observations: torch.Tensor,
        states: torch.Tensor,
        *,
        train: bool = False,
    ) -> dict[str, torch.Tensor]:
        return {
            "observations": agent._observation_preprocessor(observations, train=train),
            "states": agent._state_preprocessor(states, train=train),
        }

    def _bootstrap_last_values(self) -> torch.Tensor:
        """Evaluate critic on stored next observations (inverse value preprocessor on output)."""
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            self.value.enable_training_mode(False)
            inputs = self._preprocessed_inputs(
                self,
                self._current_next_observations.float(),
                self._current_next_states.float() if self._current_next_states is not None else self._current_next_observations.float(),
            )
            if self._rnn:
                inputs["rnn"] = self._rnn_initial_states["value"]
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            return self._value_preprocessor(last_values, inverse=True)

    def _rnn_kwargs_for_minibatch(
        self,
        batch_index: int,
        sampled_rnn_batches: List[List[torch.Tensor]],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """Build ``rnn`` kwargs for policy and value for one training minibatch."""
        if not self._rnn:
            return {}, {}
        if self.policy is self.value:
            rnn_policy = {
                "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[batch_index]],
                "terminated": terminated,
                "truncated": truncated,
            }
            return rnn_policy, rnn_policy
        rnn_policy = {
            "rnn": [
                s.transpose(0, 1)
                for s, n in zip(sampled_rnn_batches[batch_index], self._rnn_tensors_names)
                if "policy" in n
            ],
            "terminated": terminated,
            "truncated": truncated,
        }
        rnn_value = {
            "rnn": [
                s.transpose(0, 1)
                for s, n in zip(sampled_rnn_batches[batch_index], self._rnn_tensors_names)
                if "value" in n
            ],
            "terminated": terminated,
            "truncated": truncated,
        }
        return rnn_policy, rnn_value

    def _hjb_loss(
        self,
        sampled_observations_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        sampled_terminated: torch.Tensor,
    ) -> _HjbResult:
        """Scaled mean-squared reward-max HJB residual on the critic (FP32); zero if disabled."""
        if self._hjb_loss_scale <= 0.0:
            return _HjbResult(
                loss=torch.tensor(0.0, device=self.device),
                residual_abs_mean=0.0,
                value_abs_mean=0.0,
                running_cost_mean=0.0,
                lidar_grad_term_abs_mean=0.0,
            )
        if self._hjb_vec_d_index < 0 or self._hjb_vec_psi_index < 0:
            raise ValueError("hjb_vec_d_index and hjb_vec_psi_index must be non-negative.")
        with torch.autocast(device_type=self._device_type, enabled=False):
            hjb_obs_raw = sampled_observations_raw.detach().clone().float().requires_grad_(True)
            hjb_states = self._observation_preprocessor(hjb_obs_raw, train=False)
            hjb_inputs: dict[str, Any] = {
                "observations": hjb_states,
                "states": self._state_preprocessor(hjb_obs_raw, train=False),
            }
            if self._rnn:
                hjb_inputs["rnn"] = self._rnn_initial_states["value"]
                hjb_inputs["terminated"] = torch.zeros_like(sampled_terminated, dtype=torch.bool)
                hjb_inputs["truncated"] = torch.zeros_like(sampled_terminated, dtype=torch.bool)
            hjb_values, _ = self.value.act(hjb_inputs, role="value")
            if self._value_preprocessor is not None:
                hjb_values_phys = self._value_preprocessor(hjb_values, inverse=True)
            else:
                hjb_values_phys = hjb_values
            hjb_vec_raw = hjb_obs_raw[:, self._vec_start : self._vec_end]
            grad_vec_full = torch.autograd.grad(
                hjb_values.sum(),
                hjb_vec_raw,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_vec = torch.zeros_like(hjb_vec_raw) if grad_vec_full is None else grad_vec_full
            vec = hjb_vec_raw
            required_idx = max(self._hjb_vec_d_index, self._hjb_vec_psi_index)
            if self._hjb_lidar_K > 0:
                required_idx = max(required_idx, self._hjb_lidar_start + 2 * self._hjb_lidar_K - 1)
            if vec.shape[1] <= required_idx:
                raise ValueError(
                    "HJB vec indices out of range for current vec layout: "
                    f"vec_dim={vec.shape[1]}, required_max_idx={required_idx}, "
                    f"d_idx={self._hjb_vec_d_index}, psi_idx={self._hjb_vec_psi_index}, "
                    f"lidar_start={self._hjb_lidar_start}, lidar_K={self._hjb_lidar_K}"
                )
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
                v_ctrl = (
                    self._hjb_progress_weight * torch.cos(psi_err) + dVdd * torch.sin(psi_err)
                ) / (2.0 * control_w_t)
                w_ctrl = dVdpsi / (0.2 * control_w_t)
            else:
                clamped_actions = torch.clamp(sampled_actions[:, 0:2].float(), min=-1.0, max=1.0)
                v_ctrl = clamped_actions[:, 0:1] * self._hjb_max_lin_vel
                w_ctrl = clamped_actions[:, 1:2] * self._hjb_max_ang_vel

            d_dot = v_ctrl * torch.sin(psi_err)
            psi_dot = w_ctrl
            lidar_grad_term = torch.zeros_like(d_err)
            if self._hjb_lidar_K > 0:
                start = self._hjb_lidar_start
                end = start + 2 * self._hjb_lidar_K
                lidar_xy = vec[:, start:end].reshape(-1, self._hjb_lidar_K, 2)
                grad_lidar = grad_vec[:, start:end].reshape(-1, self._hjb_lidar_K, 2)
                x_k = lidar_xy[:, :, 0]
                y_k = lidar_xy[:, :, 1]
                dV_dx = grad_lidar[:, :, 0]
                dV_dy = grad_lidar[:, :, 1]
                x_dot_k = -v_ctrl + w_ctrl * y_k
                y_dot_k = -w_ctrl * x_k
                lidar_grad_term = (dV_dx * x_dot_k + dV_dy * y_dot_k).mean(dim=1, keepdim=True)
            control_cost = v_ctrl * v_ctrl + 0.1 * (w_ctrl * w_ctrl)
            running_cost = (
                self._hjb_time_weight
                + self._hjb_distance_weight * (d_err * d_err)
                + self._hjb_heading_weight * (1.0 - torch.cos(psi_err))
                + control_w_t * control_cost
                - self._hjb_progress_weight * (v_ctrl * torch.cos(psi_err))
            )
            hamiltonian = (
                -running_cost
                + dVdd * d_dot
                + dVdpsi * psi_dot
                + lidar_grad_term
                - self._hjb_discount_rate * hjb_values_phys
            )
            return _HjbResult(
                loss=self._hjb_loss_scale * torch.mean(hamiltonian * hamiltonian),
                residual_abs_mean=float(torch.mean(torch.abs(hamiltonian)).detach().item()),
                value_abs_mean=float(torch.mean(torch.abs(hjb_values_phys)).detach().item()),
                running_cost_mean=float(torch.mean(running_cost).detach().item()),
                lidar_grad_term_abs_mean=float(
                    torch.mean(torch.abs(lidar_grad_term)).detach().item()
                ),
            )

    def _minibatch_losses(
        self,
        epoch: int,
        sampled_observations_raw: torch.Tensor,
        sampled_states_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        sampled_terminated: torch.Tensor,
        sampled_truncated: torch.Tensor,
        sampled_log_prob: torch.Tensor,
        sampled_values: torch.Tensor,
        sampled_returns: torch.Tensor,
        sampled_advantages: torch.Tensor,
        rnn_policy: dict,
        rnn_value: dict,
    ) -> _MinibatchLosses:
        """Forward PPO policy/value and HJB; return losses and KL for one minibatch."""
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = self._preprocessed_inputs(
                self,
                sampled_observations_raw,
                sampled_states_raw,
                train=not epoch,
            )
            _, outputs = self.policy.act(
                {**inputs, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
            )
            next_log_prob = outputs["log_prob"]

            with torch.no_grad():
                ratio_kl = next_log_prob - sampled_log_prob
                kl_divergence = ((torch.exp(ratio_kl) - 1) - ratio_kl).mean()

            if self.cfg.entropy_loss_scale:
                entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
            else:
                entropy_loss = torch.tensor(0.0, device=self.device)

            ratio = torch.exp(next_log_prob - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * torch.clip(
                ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
            )
            policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

            predicted_values, _ = self.value.act({**inputs, **rnn_value}, role="value")
            if self.cfg.value_clip > 0:
                predicted_values = sampled_values + torch.clip(
                    predicted_values - sampled_values,
                    min=-self.cfg.value_clip,
                    max=self.cfg.value_clip,
                )
            value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

            hjb = self._hjb_loss(sampled_observations_raw, sampled_actions, sampled_terminated)

        return _MinibatchLosses(
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            value_loss=value_loss,
            hjb_loss=hjb.loss,
            kl_divergence=kl_divergence,
            hjb_residual_abs_mean=hjb.residual_abs_mean,
            hjb_value_abs_mean=hjb.value_abs_mean,
            hjb_running_cost_mean=hjb.running_cost_mean,
            hjb_lidar_grad_term_abs_mean=hjb.lidar_grad_term_abs_mean,
        )

    def _optimizer_step(self, total_loss: torch.Tensor) -> None:
        """Backward on combined losses, distributed grad sync, optional clip, optimizer step."""
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

        if config.torch.is_distributed:
            self.policy.reduce_parameters()
            if self.policy is not self.value:
                self.value.reduce_parameters()

        if self.cfg.grad_norm_clip > 0:
            self.scaler.unscale_(self.optimizer)
            if self.policy is self.value:
                params = self.policy.parameters()
            else:
                params = itertools.chain(self.policy.parameters(), self.value.parameters())
            nn.utils.clip_grad_norm_(params, self.cfg.grad_norm_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _schedule_learning_rate(self, kl_divergences: List[torch.Tensor]) -> None:
        """Step LR scheduler once per epoch (KL-adaptive uses mean KL over minibatches)."""
        if self.scheduler is None:
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
        cumulative_hjb_loss: float,
        cumulative_hjb_residual_abs_mean: float,
        cumulative_hjb_value_abs_mean: float,
        cumulative_hjb_running_cost_mean: float,
        cumulative_hjb_lidar_grad_term_abs_mean: float,
        denom: float,
    ) -> None:
        """Write averaged losses and policy stats to experiment tracking."""
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)
        if self._hjb_loss_scale > 0.0:
            self.track_data("Loss / HJB loss", cumulative_hjb_loss / denom)
            self.track_data("HJB / residual abs mean", cumulative_hjb_residual_abs_mean / denom)
            self.track_data("HJB / value phys abs mean", cumulative_hjb_value_abs_mean / denom)
            self.track_data("HJB / running cost mean", cumulative_hjb_running_cost_mean / denom)
            if self._hjb_lidar_K > 0:
                self.track_data(
                    "HJB / lidar grad term abs mean",
                    cumulative_hjb_lidar_grad_term_abs_mean / denom,
                )
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler is not None:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def _update(self, timestep: int, timesteps: int) -> None:
        """One PPO-RNN update with optional HJB critic regularizer."""
        last_values = self._bootstrap_last_values()
        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(
            names=self._tensors_names,
            mini_batches=self.cfg.mini_batches,
            sequence_length=self._rnn_sequence_length,
        )
        sampled_rnn_batches: List[List[torch.Tensor]] = []
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self.cfg.mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_hjb_loss = 0.0
        cumulative_hjb_residual_abs_mean = 0.0
        cumulative_hjb_value_abs_mean = 0.0
        cumulative_hjb_running_cost_mean = 0.0
        cumulative_hjb_lidar_grad_term_abs_mean = 0.0

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences: List[torch.Tensor] = []
            for i, (
                sampled_observations_raw,
                sampled_states_raw,
                sampled_actions,
                sampled_terminated,
                sampled_truncated,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in enumerate(sampled_batches):
                rnn_policy, rnn_value = self._rnn_kwargs_for_minibatch(
                    i, sampled_rnn_batches, sampled_terminated, sampled_truncated
                )
                losses = self._minibatch_losses(
                    epoch,
                    sampled_observations_raw,
                    sampled_states_raw,
                    sampled_actions,
                    sampled_terminated,
                    sampled_truncated,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                    rnn_policy,
                    rnn_value,
                )
                kl_divergences.append(losses.kl_divergence)

                if self.cfg.kl_threshold and losses.kl_divergence > self.cfg.kl_threshold:
                    break

                total = losses.policy_loss + losses.entropy_loss + losses.value_loss + losses.hjb_loss
                self._optimizer_step(total)

                cumulative_policy_loss += float(losses.policy_loss.item())
                cumulative_value_loss += float(losses.value_loss.item())
                cumulative_hjb_loss += float(losses.hjb_loss.item())
                cumulative_hjb_residual_abs_mean += losses.hjb_residual_abs_mean
                cumulative_hjb_value_abs_mean += losses.hjb_value_abs_mean
                cumulative_hjb_running_cost_mean += losses.hjb_running_cost_mean
                cumulative_hjb_lidar_grad_term_abs_mean += losses.hjb_lidar_grad_term_abs_mean
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += float(losses.entropy_loss.item())

            self._schedule_learning_rate(kl_divergences)

        denom = float(self.cfg.learning_epochs * self.cfg.mini_batches)
        self._track_update_metrics(
            cumulative_policy_loss,
            cumulative_value_loss,
            cumulative_entropy_loss,
            cumulative_hjb_loss,
            cumulative_hjb_residual_abs_mean,
            cumulative_hjb_value_abs_mean,
            cumulative_hjb_running_cost_mean,
            cumulative_hjb_lidar_grad_term_abs_mean,
            denom,
        )
