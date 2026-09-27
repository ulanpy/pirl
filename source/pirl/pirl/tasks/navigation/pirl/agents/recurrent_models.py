from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .obs_layout import get_vec_costmap_layout


def _flat_tensor(inputs: Mapping[str, Any]) -> torch.Tensor:
    """Preprocessed flat observation tensor (skrl 2.x uses ``observations``)."""
    tensor = inputs.get("observations")
    if tensor is None:
        tensor = inputs["states"]
    return tensor


class _TransformerBackbone(nn.Module):
    """Per-observation encoder followed by attention over a sliding token window.

    SKRL's recurrent state carries [context_len, batch, model_dim + 1]:
    encoded observations plus a validity bit. Zero state is an empty cache.
    Cached rollout tokens are detached; tokens encoded within a training window
    retain gradients into the CNN/MLPs.
    """

    def __init__(
        self,
        vec_dim: int,
        costmap_shape: tuple[int, int, int],
        transformer_model_dim: int,
        transformer_num_heads: int,
        transformer_num_layers: int,
        transformer_context_len: int,
        transformer_ff_dim: int,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(
            transformer_model_dim, transformer_num_heads, transformer_num_layers,
            transformer_context_len, transformer_ff_dim,
        ) < 1:
            raise ValueError("Transformer dimensions, heads, layers and context must be positive.")
        if transformer_model_dim % transformer_num_heads:
            raise ValueError("transformer_model_dim must be divisible by transformer_num_heads.")
        self.model_dim = transformer_model_dim
        self.num_heads = transformer_num_heads
        self.context_len = transformer_context_len
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
            nn.Linear(256, self.model_dim),
            nn.ELU(),
        )
        self.token_norm = nn.LayerNorm(self.model_dim)
        # Position is relative to the current tick: slots -context_len, ..., 0.
        self.position_embedding = nn.Parameter(torch.empty(self.context_len + 1, self.model_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.transformer = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=0.0,  # PPO replay must use the same policy as collection.
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(transformer_num_layers)
        )
        self.output_norm = nn.LayerNorm(self.model_dim)

    def _attend(
        self, seq: torch.Tensor, cache: torch.Tensor, done: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate exactly the same bounded windows in replay and single-step use."""
        batch, steps, _ = seq.shape
        past = cache.transpose(0, 1).to(dtype=seq.dtype)
        tokens = torch.cat((past[..., :-1], seq), dim=1)
        valid = torch.cat((past[..., -1].bool(), torch.ones_like(done)), dim=1)
        # done[t] describes the transition AFTER observation t. Reset before t+1.
        episode = done.long().cumsum(dim=1) - done.long()
        token_episode = torch.cat((episode.new_zeros(batch, self.context_len), episode), dim=1)
        window_size = self.context_len + 1
        windows = tokens.unfold(1, window_size, 1).permute(0, 1, 3, 2)
        window_valid = valid.unfold(1, window_size, 1) & (
            token_episode.unfold(1, window_size, 1) == episode.unsqueeze(-1)
        )
        windows = windows.reshape(batch * steps, window_size, self.model_dim)
        window_valid = window_valid.reshape(batch * steps, window_size)
        windows = windows.masked_fill(~window_valid.unsqueeze(-1), 0)
        windows = windows + self.position_embedding.to(dtype=seq.dtype)

        causal = torch.ones(window_size, window_size, dtype=torch.bool, device=seq.device).triu(1)
        mask = causal.unsqueeze(0) | ~window_valid.unsqueeze(1)
        # Padded queries may attend to themselves so no softmax row is all masked.
        # They remain blocked as keys for every valid query in every layer.
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)
        mask = mask.repeat_interleave(self.num_heads, dim=0)
        for layer in self.transformer:
            windows = layer(windows, src_mask=mask)
        features = self.output_norm(windows[:, -1])

        next_valid = valid[:, -self.context_len:] & (
            token_episode[:, -self.context_len:] == done.long().sum(dim=1, keepdim=True)
        )
        next_tokens = tokens[:, -self.context_len:].masked_fill(~next_valid.unsqueeze(-1), 0)
        next_cache = torch.cat((next_tokens, next_valid.unsqueeze(-1).to(seq.dtype)), dim=-1)
        return features, next_cache.transpose(0, 1).contiguous()

    def forward(
        self,
        states: torch.Tensor,
        cache: torch.Tensor | None,
        sequence_length: int,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
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
        enc = self.token_norm(self.fusion(enc))
        steps = sequence_length if terminated is not None or truncated is not None else 1
        if steps < 1 or enc.shape[0] % steps:
            raise ValueError(
                f"Batch size {enc.shape[0]} must be divisible by positive sequence_length={steps}."
            )
        batch = enc.shape[0] // steps
        if cache is None:
            cache = enc.new_zeros(self.context_len, batch, self.model_dim + 1)
        elif cache.shape[1] == enc.shape[0]:
            # SKRL stores an incoming cache per transition. Replay starts from
            # the first cache of each sequence, rebuilding later tokens with gradients.
            cache = cache[:, ::steps, :]
        if tuple(cache.shape) != (self.context_len, batch, self.model_dim + 1):
            raise ValueError(f"Unexpected transformer cache shape: {tuple(cache.shape)}.")
        done = torch.zeros(batch, steps, dtype=torch.bool, device=enc.device)
        if terminated is not None:
            done = done | terminated.reshape(batch, steps).bool()
        if truncated is not None:
            done = done | truncated.reshape(batch, steps).bool()
        return self._attend(enc.reshape(batch, steps, -1), cache.detach(), done)


class RecurrentGaussianPolicy(GaussianMixin, Model):
    """Causal transformer actor; name and ``rnn`` keys retain SKRL runner compatibility."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_space=None,
        num_envs: int = 1,
        sequence_length: int = 32,
        transformer_model_dim: int = 128,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2,
        transformer_context_len: int = 32,
        transformer_ff_dim: int = 512,
        aux_dim: int = 0,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.0,
        return_source: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
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
        if self._sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        self.backbone = _TransformerBackbone(
            vec_dim=self._vec_dim,
            costmap_shape=self._costmap_shape,
            transformer_model_dim=int(transformer_model_dim),
            transformer_num_heads=int(transformer_num_heads),
            transformer_num_layers=int(transformer_num_layers),
            transformer_context_len=int(transformer_context_len),
            transformer_ff_dim=int(transformer_ff_dim),
            aux_dim=int(aux_dim),
        )
        action_dim = int(self.num_actions) if self.num_actions is not None else int(math.prod(action_space.shape))
        self.mean_head = nn.Sequential(
            nn.Linear(int(transformer_model_dim), 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )
        self.log_std_parameter = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sequence_length": self._sequence_length,
                "sizes": [(self.backbone.context_len, self._num_envs, self.backbone.model_dim + 1)],
            }
        }

    def compute(self, inputs, role=""):
        states = _flat_tensor(inputs)
        rnn_list = inputs.get("rnn", None)
        cache = rnn_list[0] if rnn_list else None
        feats, next_cache = self.backbone(
            states=states,
            cache=cache,
            sequence_length=self._sequence_length,
            terminated=inputs.get("terminated"),
            truncated=inputs.get("truncated"),
            vec_start=self._vec_start,
            vec_dim=self._vec_dim,
            costmap_start=self._costmap_start,
            costmap_shape=self._costmap_shape,
        )
        mean = self.mean_head(feats)
        return mean, {"log_std": self.log_std_parameter, "rnn": [next_cache]}


class FeedForwardDeterministicValue(DeterministicMixin, Model):
    """Deterministic value model without recurrent state (vec+costmap fusion only)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_space=None,
        clip_actions: bool = False,
        return_source: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
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
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        states = _flat_tensor(inputs)
        c, h, w = self._costmap_shape
        vec = states[:, self._vec_start : self._vec_start + self._vec_dim]
        costmap = states[:, self._costmap_start : self._costmap_start + (c * h * w)].reshape(-1, c, h, w)
        feats = torch.cat((self.vec_net(vec), self.cnn(costmap)), dim=-1)
        feats = self.fusion(feats)
        return self.value_head(feats), {}
