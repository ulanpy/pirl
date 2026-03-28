from __future__ import annotations

import math
from typing import Any, Mapping

import gymnasium
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .obs_layout import get_vec_costmap_layout


class _RecurrentBackbone(nn.Module):
    def __init__(
        self,
        vec_dim: int,
        costmap_shape: tuple[int, int, int],
        gru_hidden_size: int,
        gru_num_layers: int,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        c, h, w = costmap_shape
        self.aux_dim = min(aux_dim, vec_dim)
        core_dim = vec_dim - self.aux_dim
        self.vec_net = nn.Sequential(
            nn.Linear(core_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        if self.aux_dim > 0:
            self.aux_net = nn.Sequential(
                nn.Linear(self.aux_dim, 32),
                nn.ELU(),
                nn.Linear(32, 32),
                nn.ELU(),
            )
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_dim = int(self.cnn(dummy).shape[-1])
        fusion_input_dim = 64 + cnn_dim
        if self.aux_dim > 0:
            fusion_input_dim += 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.pre_gru_ln = nn.LayerNorm(128)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.post_gru_ln = nn.LayerNorm(gru_hidden_size)

    def forward(
        self,
        states: torch.Tensor,
        rnn_state: torch.Tensor | None,
        sequence_length: int,
        terminated: torch.Tensor | None = None,
        vec_start: int = 0,
        vec_dim: int = 0,
        costmap_start: int = 0,
        costmap_shape: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c, h, w = costmap_shape
        vec = states[:, vec_start : vec_start + vec_dim]
        costmap = states[:, costmap_start : costmap_start + (c * h * w)].reshape(-1, c, h, w)
        if self.aux_dim > 0:
            if vec.shape[-1] < self.aux_dim:
                raise ValueError(
                    f"Got vec dimension {vec.shape[-1]} smaller than aux_dim={self.aux_dim}."
                )
            core_vec = vec[:, : vec.shape[-1] - self.aux_dim]
            aux_vec = vec[:, vec.shape[-1] - self.aux_dim :]
            core_enc = self.vec_net(core_vec)
            aux_enc = self.aux_net(aux_vec)
            enc = torch.cat((core_enc, aux_enc, self.cnn(costmap)), dim=-1)
        else:
            enc = torch.cat((self.vec_net(vec), self.cnn(costmap)), dim=-1)
        enc = self.pre_gru_ln(self.fusion(enc))

        use_sequence = (
            terminated is not None
            and sequence_length > 1
            and (enc.shape[0] % sequence_length == 0)
        )

        if terminated is not None and sequence_length > 1 and (enc.shape[0] % sequence_length != 0):
            raise ValueError(
                "RNN training batch size is not divisible by sequence_length. "
                f"Got batch={enc.shape[0]}, sequence_length={sequence_length}. "
                "Adjust sequence_length / rollouts / num_envs / mini_batches so each sampled batch is divisible."
            )

        if not use_sequence:
            if rnn_state is None:
                rnn_state = torch.zeros(
                    self.gru.num_layers,
                    enc.shape[0],
                    self.gru.hidden_size,
                    device=enc.device,
                    dtype=enc.dtype,
                )
            out, rnn_next = self.gru(enc.unsqueeze(1), rnn_state)
            return self.post_gru_ln(out.squeeze(1)), rnn_next

        batch = enc.shape[0] // sequence_length
        seq = enc.reshape(batch, sequence_length, -1)

        # skrl memory stores RNN state per transition. For sequence training we need
        # initial state per sequence (take first state in each sequence window).
        if rnn_state is None:
            rnn_state = torch.zeros(
                self.gru.num_layers, batch, self.gru.hidden_size, device=enc.device, dtype=enc.dtype
            )
        elif rnn_state.shape[1] == enc.shape[0]:
            rnn_state = rnn_state[:, ::sequence_length, :]
        elif rnn_state.shape[1] != batch:
            rnn_state = rnn_state[:, :batch, :]

        assert terminated is not None
        done_mask = terminated.reshape(batch, sequence_length, -1).squeeze(-1).float()

        outputs = []
        h_t = rnn_state
        for t in range(sequence_length):
            if done_mask is not None:
                # Reset hidden state where episode ended at current transition.
                alive = (1.0 - done_mask[:, t]).view(1, batch, 1)
                h_t = h_t * alive
            o_t, h_t = self.gru(seq[:, t : t + 1, :], h_t)
            outputs.append(o_t)
        out = torch.cat(outputs, dim=1).reshape(-1, self.gru.hidden_size)
        out = self.post_gru_ln(out)
        return out, h_t


class RecurrentGaussianPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int = 1,
        sequence_length: int = 32,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        aux_dim: int = 0,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.0,
        return_source: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction="sum",
        )
        self._vec_start, self._vec_dim, self._costmap_start, self._costmap_shape = get_vec_costmap_layout(
            observation_space
        )
        self._num_envs = int(num_envs)
        self._sequence_length = int(sequence_length)
        self.backbone = _RecurrentBackbone(
            vec_dim=self._vec_dim,
            costmap_shape=self._costmap_shape,
            gru_hidden_size=int(gru_hidden_size),
            gru_num_layers=int(gru_num_layers),
            aux_dim=int(aux_dim),
        )
        action_dim = int(self.num_actions) if self.num_actions is not None else int(math.prod(action_space.shape))
        self.mean_head = nn.Linear(int(gru_hidden_size), action_dim)
        self.log_std_parameter = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sequence_length": self._sequence_length,
                "sizes": [(self.backbone.gru.num_layers, self._num_envs, self.backbone.gru.hidden_size)],
            }
        }

    def compute(self, inputs, role=""):
        states = inputs["states"]
        rnn_list = inputs.get("rnn", None)
        rnn_state = rnn_list[0] if rnn_list else None
        terminated = inputs.get("terminated", None)
        feats, rnn_next = self.backbone(
            states=states,
            rnn_state=rnn_state,
            sequence_length=self._sequence_length,
            terminated=terminated,
            vec_start=self._vec_start,
            vec_dim=self._vec_dim,
            costmap_start=self._costmap_start,
            costmap_shape=self._costmap_shape,
        )
        mean = self.mean_head(feats)
        return mean, self.log_std_parameter, {"rnn": [rnn_next]}


class RecurrentDeterministicValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int = 1,
        sequence_length: int = 32,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        aux_dim: int = 0,
        clip_actions: bool = False,
        return_source: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._vec_start, self._vec_dim, self._costmap_start, self._costmap_shape = get_vec_costmap_layout(
            observation_space
        )
        self._num_envs = int(num_envs)
        self._sequence_length = int(sequence_length)
        self.backbone = _RecurrentBackbone(
            vec_dim=self._vec_dim,
            costmap_shape=self._costmap_shape,
            gru_hidden_size=int(gru_hidden_size),
            gru_num_layers=int(gru_num_layers),
            aux_dim=int(aux_dim),
        )
        self.value_head = nn.Linear(int(gru_hidden_size), 1)

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sequence_length": self._sequence_length,
                "sizes": [(self.backbone.gru.num_layers, self._num_envs, self.backbone.gru.hidden_size)],
            }
        }

    def compute(self, inputs, role=""):
        states = inputs["states"]
        rnn_list = inputs.get("rnn", None)
        rnn_state = rnn_list[0] if rnn_list else None
        terminated = inputs.get("terminated", None)
        feats, rnn_next = self.backbone(
            states=states,
            rnn_state=rnn_state,
            sequence_length=self._sequence_length,
            terminated=terminated,
            vec_start=self._vec_start,
            vec_dim=self._vec_dim,
            costmap_start=self._costmap_start,
            costmap_shape=self._costmap_shape,
        )
        return self.value_head(feats), {"rnn": [rnn_next]}


class FeedForwardDeterministicValue(DeterministicMixin, Model):
    """Deterministic value model without recurrent state (vec+costmap fusion only)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        return_source: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._vec_start, self._vec_dim, self._costmap_start, self._costmap_shape = get_vec_costmap_layout(
            observation_space
        )
        c, h, w = self._costmap_shape
        self.vec_net = nn.Sequential(
            nn.Linear(self._vec_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_dim = int(self.cnn(dummy).shape[-1])
        self.fusion = nn.Sequential(
            nn.Linear(64 + cnn_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.value_head = nn.Linear(128, 1)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        c, h, w = self._costmap_shape
        vec = states[:, self._vec_start : self._vec_start + self._vec_dim]
        costmap = states[:, self._costmap_start : self._costmap_start + (c * h * w)].reshape(-1, c, h, w)
        feats = torch.cat((self.vec_net(vec), self.cnn(costmap)), dim=-1)
        feats = self.fusion(feats)
        return self.value_head(feats), {}
