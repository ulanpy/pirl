"""PPO-RNN agent with optional HJB PINN-style regularizer on the critic."""

import copy
import itertools
import math
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.resources.schedulers.torch import KLAdaptiveLR

from .obs_layout import get_vec_costmap_layout


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


# Merged into skrl PPO_DEFAULT_CONFIG; overridden by YAML ``agent:`` section.
PPOHjbRNN_default_config = {
    # HJB regularizer (PINN-style) on value function. If 0.0, HJB branch is disabled.
    "hjb_loss_scale": 0.5,
    # Differential-drive kinematics scales for normalized actions.
    "hjb_max_lin_vel": 0.5,
    "hjb_max_ang_vel": 1.5,
    # Running cost weights in Hamiltonian.
    "hjb_time_weight": 0.5,
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
    # ObservationSchemaV2.1 vec layout:
    # [vx, wz, d_signed, heading_error, path_obs..., lidar_sector_xy..., prev_action, prev_reward]
    "hjb_vec_d_index": 2,
    "hjb_vec_psi_index": 3,
    # Env step (seconds). Used to derive continuous-time discount rho = -ln(gamma)/dt
    # for the reward-max HJB residual r + grad V . f - rho V. Default matches
    # PirlEnvCfg: decimation=2 * sim.dt=1/120 => dt = 1/60 s.
    "hjb_step_dt": 1.0 / 60.0,
    # Optional LiDAR-sector hit positions used to extend HJB dynamics (Schema V2.1).
    # When ``hjb_lidar_sector_count > 0``, HJB Hamiltonian is augmented with
    # ``sum_k (dV/dx_k * x_dot_k + dV/dy_k * y_dot_k)`` for each sector hit at vec slot
    # ``[hjb_lidar_hits_start_index : start + 2 * sector_count]``.
    # Body-frame static-obstacle kinematics: x_dot = -v + w*y, y_dot = -w*x.
    "hjb_lidar_hits_start_index": -1,
    "hjb_lidar_sector_count": 0,
}


class PPOHjbRNN(PPO_RNN):
    """PPO-RNN (skrl) extended with an optional HJB-style critic penalty.

    HJB branch (if ``hjb_loss_scale > 0``) penalizes a local Hamiltonian residual built
    from selected ``vec`` indices and analytic relative kinematics; see
    ``docs/HJB_THEORY_TIME_DISTANCE.md``.
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
        """Merge defaults with ``PPOHjbRNN_default_config`` and ``cfg``."""
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(PPOHjbRNN_default_config)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # HJB config
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
        self._hjb_vec_d_index = int(self.cfg.get("hjb_vec_d_index", 2))
        self._hjb_vec_psi_index = int(self.cfg.get("hjb_vec_psi_index", 3))
        self._hjb_lidar_start = int(self.cfg.get("hjb_lidar_hits_start_index", -1))
        self._hjb_lidar_K = int(self.cfg.get("hjb_lidar_sector_count", 0))
        if self._hjb_lidar_K > 0 and self._hjb_lidar_start < 0:
            raise ValueError(
                "hjb_lidar_sector_count > 0 requires a non-negative hjb_lidar_hits_start_index."
            )
        # Continuous-time discount rate matched to PPO's per-step discount.
        # PPO learns V(x) = E[sum gamma^t r_t]; the reward-max HJB Bellman at stationarity
        # is  r + grad V . f - rho V = 0  with rho = -ln(gamma)/dt. Without the rho V term
        # the residual is offset by rho*V at the optimum and scales with value magnitude,
        # causing HJB loss to explode as V grows during training.
        self._hjb_step_dt = float(self.cfg.get("hjb_step_dt", 1.0 / 60.0))
        _gamma = float(getattr(self, "_discount_factor", 0.99))
        self._hjb_discount_rate = (
            -math.log(max(min(_gamma, 1.0 - 1e-9), 1e-9)) / max(self._hjb_step_dt, 1e-9)
        )

        # Vec slice in flattened Dict(obs); same layout as recurrent_models / skrl preprocessor.
        self._vec_start, self._vec_size, _, _ = get_vec_costmap_layout(self.observation_space)
        self._vec_end = self._vec_start + self._vec_size

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

    def _hjb_loss(
        self,
        sampled_states_raw: torch.Tensor,
        sampled_actions: torch.Tensor,
        sampled_terminated: torch.Tensor,
    ) -> _HjbResult:
        """Scaled mean-squared reward-max HJB residual on the critic (FP32); zero if disabled.

        Convention: PPO learns a reward-maximization value ``V_pi(x) = E[sum gamma^t r_t]``
        with reward ``r = -l`` (negation of the running cost ``l`` modeled below). The
        matching continuous-time Bellman stationarity is

            r(x, u*) + grad_x V(x) . f(x, u*) - rho V(x) = 0,    rho = -ln(gamma)/dt,

        so we minimize the squared residual of this identity. Using reward-max (rather
        than the cost-min form) ensures HJB gradients do not fight PPO's TD targets:
        both drive V in the same direction as policy improves.
        """
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
            # получение физических значений состояний из сырых наблюдений без нормализации
            hjb_states_raw = sampled_states_raw.detach().clone().float().requires_grad_(True)
            hjb_states = self._state_preprocessor(hjb_states_raw, train=False)
            hjb_rnn_value: dict = {}
            if self._rnn:
                hjb_rnn_value = {
                    "rnn": self._rnn_initial_states["value"],
                    "terminated": torch.zeros_like(sampled_terminated, dtype=torch.bool),
                }
            hjb_values, _, _ = self.value.act({"states": hjb_states, **hjb_rnn_value}, role="value")
            # Map V back to physical reward units so rho * V is on the same scale as the
            # reward-rate running cost. The value preprocessor is a running standardizer
            # (RunningStandardScaler) applied to TD targets, so `value.act` returns
            # normalized V; the inverse pass restores mean/std-adjusted absolute V.
            if self._value_preprocessor is not None:
                hjb_values_phys = self._value_preprocessor(hjb_values, inverse=True)
            else:
                hjb_values_phys = hjb_values
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
            # HJB state: x = [d, psi] (signed cross-track error, heading error).
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
                # Reward-max argmax of H_r = -l + grad V . f solved from dH_r/du = 0:
                #   dH_r/dv = -(2 w_u v - w_p cos psi) + dV/dd sin psi = 0
                #     => v* = (w_p cos psi + dV/dd sin psi) / (2 w_u)
                #   dH_r/dw = -(0.2 w_u w)              + dV/dpsi     = 0
                #     => w* =  dV/dpsi / (0.2 w_u)
                v_ctrl = (
                    self._hjb_progress_weight * torch.cos(psi_err) + dVdd * torch.sin(psi_err)
                ) / (2.0 * control_w_t)
                w_ctrl = dVdpsi / (0.2 * control_w_t)
            else:
                # Policy-evaluated Hamiltonian (on-policy residual).
                # PPO stores raw Gaussian samples in memory, but the env clamps them to
                # [-1, 1] before applying to wheels (see PirlEnv._pre_physics_step). Match
                # the realized physical control here so HJB residual reflects what the
                # robot actually executed instead of unbounded policy tails.
                clamped_actions = torch.clamp(sampled_actions[:, 0:2].float(), min=-1.0, max=1.0)
                v_ctrl = clamped_actions[:, 0:1] * self._hjb_max_lin_vel
                w_ctrl = clamped_actions[:, 1:2] * self._hjb_max_ang_vel

            # Small-curvature local kinematics: d_dot = v sin psi, psi_dot = w.
            d_dot = v_ctrl * torch.sin(psi_err)
            psi_dot = w_ctrl
            # Optional Schema V2.1 extension: per-sector LiDAR hit positions in body frame
            # are slotted into vec at [start, start + 2K). For a static obstacle expressed
            # in the robot body frame:  x_dot_k = -v + w * y_k,  y_dot_k = -w * x_k.
            # Their contribution to grad V . f is sum_k (dV/dx_k * x_dot_k + dV/dy_k * y_dot_k).
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
                # Mean over sectors keeps HJB scale approximately K-invariant
                # (avoids over-conservative behavior when sector_count grows).
                lidar_grad_term = (dV_dx * x_dot_k + dV_dy * y_dot_k).mean(dim=1, keepdim=True)
            # Running cost model (acts as a physics-informed prior on V shape):
            # l = w_t + w_d d^2 + w_psi (1 - cos psi) + w_u (v^2 + 0.1 w^2) - w_p v cos psi.
            control_cost = v_ctrl * v_ctrl + 0.1 * (w_ctrl * w_ctrl)
            running_cost = (
                self._hjb_time_weight
                + self._hjb_distance_weight * (d_err * d_err)
                + self._hjb_heading_weight * (1.0 - torch.cos(psi_err))
                + control_w_t * control_cost
                - self._hjb_progress_weight * (v_ctrl * torch.cos(psi_err))
            )
            # Reward-max continuous-time Bellman residual: r + grad V . f - rho V
            # with r = -l. rho V anchors the residual to zero at the optimum; without
            # it the residual is offset by rho*V and blows up as |V| grows.
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
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            sampled_states = self._state_preprocessor(sampled_states_raw, train=not epoch)
            _, next_log_prob, _ = self.policy.act(
                {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
            )

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

            hjb = self._hjb_loss(sampled_states_raw, sampled_actions, sampled_terminated)

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
        cumulative_hjb_loss: float,
        cumulative_hjb_residual_abs_mean: float,
        cumulative_hjb_value_abs_mean: float,
        cumulative_hjb_running_cost_mean: float,
        cumulative_hjb_lidar_grad_term_abs_mean: float,
        denom: float,
    ) -> None:
        """Write averaged losses and policy stats to experiment tracking (e.g. TensorBoard)."""
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self._entropy_loss_scale:
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
        cumulative_hjb_loss = 0.0
        cumulative_hjb_residual_abs_mean = 0.0
        cumulative_hjb_value_abs_mean = 0.0
        cumulative_hjb_running_cost_mean = 0.0
        cumulative_hjb_lidar_grad_term_abs_mean = 0.0

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
                    rnn_policy,
                    rnn_value,
                )
                kl_divergences.append(L.kl_divergence)

                if self._kl_threshold and L.kl_divergence > self._kl_threshold:
                    break

                total = L.policy_loss + L.entropy_loss + L.value_loss + L.hjb_loss
                self._optimizer_step(total)

                cumulative_policy_loss += float(L.policy_loss.item())
                cumulative_value_loss += float(L.value_loss.item())
                cumulative_hjb_loss += float(L.hjb_loss.item())
                cumulative_hjb_residual_abs_mean += L.hjb_residual_abs_mean
                cumulative_hjb_value_abs_mean += L.hjb_value_abs_mean
                cumulative_hjb_running_cost_mean += L.hjb_running_cost_mean
                cumulative_hjb_lidar_grad_term_abs_mean += L.hjb_lidar_grad_term_abs_mean
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(L.entropy_loss.item())

            self._schedule_learning_rate(kl_divergences)

        denom = float(self._learning_epochs * self._mini_batches)
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
