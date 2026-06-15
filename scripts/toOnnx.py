#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false
"""Export the PIRL recurrent actor policy checkpoint to ONNX.

This script reconstructs the current PIRL recurrent actor architecture from the
repository defaults and exports a deployment-oriented ONNX graph with explicit:

- input `vec`:         [B, vec_dim]
- input `costmap`:     [B, C, H, W]
- input `rnn_state`:   [num_layers, B, hidden_size]
- output `mean`:       [B, action_dim]
- output `rnn_state_out`: [num_layers, B, hidden_size]

The exported model is intended for controller-side deterministic inference
using the actor mean action and explicit GRU hidden-state carry-over.

The ONNX graph embeds the saved SKRL ``RunningStandardScaler`` state by default.
Controller code should feed deployment observations in ObservationSchemaV2.1:
``vec`` (68 floats: ego + tracking + path window + 16 LiDAR sector hits + memory)
and a Nav2-style ``costmap`` encoded as cost + known-mask history channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


class _RecurrentBackbone(torch.nn.Module):
    """Standalone copy of the PIRL recurrent actor backbone."""

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
        self.vec_net = torch.nn.Sequential(
            torch.nn.Linear(core_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
        )
        if self.aux_dim > 0:
            self.aux_net = torch.nn.Sequential(
                torch.nn.Linear(self.aux_dim, 32),
                torch.nn.ELU(),
                torch.nn.Linear(32, 32),
                torch.nn.ELU(),
            )
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(c, 16, kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_dim = int(self.cnn(dummy).shape[-1])
        fusion_input_dim = 64 + cnn_dim
        if self.aux_dim > 0:
            fusion_input_dim += 32
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
        )
        self.pre_gru_ln = torch.nn.LayerNorm(128)
        self.gru = torch.nn.GRU(
            input_size=128,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.post_gru_ln = torch.nn.LayerNorm(gru_hidden_size)

    def forward(
        self,
        states: torch.Tensor,
        rnn_state: torch.Tensor | None,
        vec_dim: int,
        costmap_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c, h, w = costmap_shape
        costmap_flat_dim = c * h * w
        costmap = states[:, :costmap_flat_dim].reshape(-1, c, h, w)
        vec = states[:, costmap_flat_dim : costmap_flat_dim + vec_dim]
        if self.aux_dim > 0:
            core_vec = vec[:, : vec.shape[-1] - self.aux_dim]
            aux_vec = vec[:, vec.shape[-1] - self.aux_dim :]
            enc = torch.cat((self.vec_net(core_vec), self.aux_net(aux_vec), self.cnn(costmap)), dim=-1)
        else:
            enc = torch.cat((self.vec_net(vec), self.cnn(costmap)), dim=-1)
        enc = self.pre_gru_ln(self.fusion(enc))

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


class RecurrentGaussianPolicy(torch.nn.Module):
    """Standalone copy of the PIRL recurrent actor module."""

    def __init__(
        self,
        vec_dim: int,
        costmap_shape: tuple[int, int, int],
        action_dim: int,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        self._vec_dim = int(vec_dim)
        self._costmap_shape = tuple(int(v) for v in costmap_shape)
        self.backbone = _RecurrentBackbone(
            vec_dim=self._vec_dim,
            costmap_shape=self._costmap_shape,
            gru_hidden_size=int(gru_hidden_size),
            gru_num_layers=int(gru_num_layers),
            aux_dim=int(aux_dim),
        )
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(int(gru_hidden_size), 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, int(action_dim)),
        )
        self.log_std_parameter = torch.nn.Parameter(torch.full((int(action_dim),), -1.0))

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
        del role
        states = inputs["states"]
        rnn_list = inputs.get("rnn", None)
        rnn_state = rnn_list[0] if rnn_list else None
        feats, rnn_next = self.backbone(
            states=states,
            rnn_state=rnn_state,
            vec_dim=self._vec_dim,
            costmap_shape=self._costmap_shape,
        )
        mean = self.mean_head(feats)
        return mean, self.log_std_parameter, {"rnn": [rnn_next]}


def strip_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor] | None:
    if not state_dict:
        return None
    if not all(key.startswith(prefix) for key in state_dict):
        return None
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def extract_policy_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    """Extract the recurrent policy state dict from common skrl checkpoint layouts."""
    if isinstance(checkpoint_obj, dict):
        direct_candidate = checkpoint_obj.get("policy")
        if isinstance(direct_candidate, dict) and direct_candidate:
            return direct_candidate

        for outer_key in ("agent", "models", "modules", "state_dict", "checkpoint"):
            outer = checkpoint_obj.get(outer_key)
            if not isinstance(outer, dict):
                continue
            direct_candidate = outer.get("policy")
            if isinstance(direct_candidate, dict) and direct_candidate:
                return direct_candidate

        flat_tensor_dict = {
            key: value
            for key, value in checkpoint_obj.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        if flat_tensor_dict:
            for prefix in ("policy.", "models.policy.", "agent.policy.", "module.policy."):
                stripped = strip_prefix_from_state_dict(flat_tensor_dict, prefix)
                if stripped:
                    return stripped

            # Fallback: assume the checkpoint itself is already a policy state dict.
            if any(key.startswith("backbone.") or key.startswith("mean_head.") for key in flat_tensor_dict):
                return flat_tensor_dict

    raise RuntimeError(
        "Could not extract policy state_dict from checkpoint. "
        "Expected a skrl checkpoint containing a `policy` module."
    )


def extract_state_preprocessor_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    """Extract the saved SKRL RunningStandardScaler state from a checkpoint."""
    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError("Expected a dict checkpoint containing `state_preprocessor`.")

    candidate = checkpoint_obj.get("state_preprocessor")
    if isinstance(candidate, dict) and candidate:
        return candidate

    for outer_key in ("agent", "modules", "state_dict", "checkpoint"):
        outer = checkpoint_obj.get(outer_key)
        if not isinstance(outer, dict):
            continue
        candidate = outer.get("state_preprocessor")
        if isinstance(candidate, dict) and candidate:
            return candidate

    raise RuntimeError(
        "Could not extract `state_preprocessor` from checkpoint. "
        "Use --skip-state-normalization only if normalization is applied outside ONNX."
    )


class ExportableRecurrentPolicy(torch.nn.Module):
    """Deployment wrapper with explicit vec/costmap/rnn inputs and mean/rnn outputs.

    This wrapper exports a single recurrent step using manual GRU equations from
    the trained PyTorch GRU weights. That avoids exporting the ONNX `GRU` op,
    which is the most likely source of the ORT load failure on the robot side.
    """

    def __init__(
        self,
        policy: RecurrentGaussianPolicy,
        state_preprocessor_state_dict: dict[str, torch.Tensor] | None,
        scaler_epsilon: float,
        scaler_clip_threshold: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.scaler_epsilon = float(scaler_epsilon)
        self.scaler_clip_threshold = float(scaler_clip_threshold)

        c, h, w = self.policy._costmap_shape
        flat_dim = (c * h * w) + self.policy._vec_dim
        if state_preprocessor_state_dict is None:
            running_mean = torch.zeros(flat_dim, dtype=torch.float32)
            running_variance = torch.ones(flat_dim, dtype=torch.float32)
        else:
            running_mean = state_preprocessor_state_dict["running_mean"].float()
            running_variance = state_preprocessor_state_dict["running_variance"].float()
            if running_mean.numel() != flat_dim or running_variance.numel() != flat_dim:
                raise ValueError(
                    "State preprocessor size mismatch. "
                    f"Expected {flat_dim}, got mean={running_mean.numel()} variance={running_variance.numel()}."
                )

        self.register_buffer("state_mean", running_mean.reshape(1, -1))
        self.register_buffer("state_std", torch.sqrt(running_variance).reshape(1, -1))

    @staticmethod
    def _gru_cell(
        x: torch.Tensor,
        h: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> torch.Tensor:
        gi = torch.nn.functional.linear(x, weight_ih, bias_ih)
        gh = torch.nn.functional.linear(h, weight_hh, bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + (reset_gate * h_n))
        return new_gate + update_gate * (h - new_gate)

    def forward(
        self,
        vec: torch.Tensor,
        costmap: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = vec.shape[0]
        flat_costmap = costmap.reshape(batch, -1)
        # skrl Dict flattening uses sorted key order: "costmap", then "vec".
        flat_states = torch.cat((flat_costmap, vec), dim=-1)
        flat_states = torch.clamp(
            (flat_states - self.state_mean) / (self.state_std + self.scaler_epsilon),
            min=-self.scaler_clip_threshold,
            max=self.scaler_clip_threshold,
        )
        vec_dim = self.policy._vec_dim
        c, h, w = self.policy._costmap_shape
        costmap_flat_dim = c * h * w
        costmap_tensor = flat_states[:, :costmap_flat_dim].reshape(-1, c, h, w)
        vec_tensor = flat_states[:, costmap_flat_dim : costmap_flat_dim + vec_dim]

        if self.policy.backbone.aux_dim > 0:
            core_vec = vec_tensor[:, : vec_tensor.shape[-1] - self.policy.backbone.aux_dim]
            aux_vec = vec_tensor[:, vec_tensor.shape[-1] - self.policy.backbone.aux_dim :]
            enc = torch.cat(
                (
                    self.policy.backbone.vec_net(core_vec),
                    self.policy.backbone.aux_net(aux_vec),
                    self.policy.backbone.cnn(costmap_tensor),
                ),
                dim=-1,
            )
        else:
            enc = torch.cat(
                (
                    self.policy.backbone.vec_net(vec_tensor),
                    self.policy.backbone.cnn(costmap_tensor),
                ),
                dim=-1,
            )
        enc = self.policy.backbone.pre_gru_ln(self.policy.backbone.fusion(enc))

        hidden_layers = []
        gru = self.policy.backbone.gru
        layer_input = enc
        for layer_idx in range(gru.num_layers):
            layer_hidden = rnn_state[layer_idx]
            layer_output = self._gru_cell(
                x=layer_input,
                h=layer_hidden,
                weight_ih=getattr(gru, f"weight_ih_l{layer_idx}"),
                weight_hh=getattr(gru, f"weight_hh_l{layer_idx}"),
                bias_ih=getattr(gru, f"bias_ih_l{layer_idx}"),
                bias_hh=getattr(gru, f"bias_hh_l{layer_idx}"),
            )
            hidden_layers.append(layer_output)
            layer_input = layer_output

        next_rnn_state = torch.stack(hidden_layers, dim=0)
        feats = self.policy.backbone.post_gru_ln(layer_input)
        mean = self.policy.mean_head(feats)
        return mean, next_rnn_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PIRL recurrent actor checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to the PyTorch/skrl checkpoint file.")
    parser.add_argument("--output", required=True, help="Path to the output ONNX file.")
    parser.add_argument("--vec-dim", type=int, default=68, help="Vector observation dimension.")
    parser.add_argument("--costmap-channels", type=int, default=6, help="Costmap channel count.")
    parser.add_argument("--costmap-cells", type=int, default=100, help="Costmap width/height in cells.")
    parser.add_argument("--action-dim", type=int, default=2, help="Action dimension.")
    parser.add_argument("--gru-hidden-size", type=int, default=256, help="GRU hidden size.")
    parser.add_argument("--gru-num-layers", type=int, default=1, help="GRU layer count.")
    parser.add_argument("--aux-dim", type=int, default=8, help="Auxiliary vector tail dimension.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Export batch size. Keep this at 1 for controller deployment.",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch axes.")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version.")
    parser.add_argument(
        "--skip-state-normalization",
        action="store_true",
        help="Do not embed the checkpoint's RunningStandardScaler in ONNX.",
    )
    parser.add_argument("--scaler-epsilon", type=float, default=1.0e-8, help="RunningStandardScaler epsilon.")
    parser.add_argument(
        "--scaler-clip-threshold",
        type=float,
        default=5.0,
        help="RunningStandardScaler output clipping threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.dynamic_batch and args.batch_size != 1:
        raise ValueError("--dynamic-batch expects export with --batch-size 1")

    policy = RecurrentGaussianPolicy(
        vec_dim=args.vec_dim,
        costmap_shape=(args.costmap_channels, args.costmap_cells, args.costmap_cells),
        action_dim=args.action_dim,
        gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers,
        aux_dim=args.aux_dim,
    )
    policy.eval()

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state_dict = extract_policy_state_dict(checkpoint_obj)
    load_result = policy.load_state_dict(policy_state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Policy checkpoint mismatch.\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    state_preprocessor_state_dict = None
    if not args.skip_state_normalization:
        state_preprocessor_state_dict = extract_state_preprocessor_state_dict(checkpoint_obj)

    export_model = ExportableRecurrentPolicy(
        policy=policy,
        state_preprocessor_state_dict=state_preprocessor_state_dict,
        scaler_epsilon=args.scaler_epsilon,
        scaler_clip_threshold=args.scaler_clip_threshold,
    ).eval()

    dummy_vec = torch.zeros((args.batch_size, args.vec_dim), dtype=torch.float32)
    dummy_costmap = torch.zeros(
        (args.batch_size, args.costmap_channels, args.costmap_cells, args.costmap_cells),
        dtype=torch.float32,
    )
    dummy_rnn_state = torch.zeros(
        (args.gru_num_layers, args.batch_size, args.gru_hidden_size),
        dtype=torch.float32,
    )

    export_kwargs: dict[str, Any] = {}
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "vec": {0: "batch"},
            "costmap": {0: "batch"},
            "rnn_state": {1: "batch"},
            "mean": {0: "batch"},
            "rnn_state_out": {1: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy_vec, dummy_costmap, dummy_rnn_state),
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["vec", "costmap", "rnn_state"],
            output_names=["mean", "rnn_state_out"],
            **export_kwargs,
        )

    normalization = "embedded" if state_preprocessor_state_dict is not None else "external/disabled"
    print(f"Exported ONNX policy to: {output_path}")
    print(
        "Inputs: vec "
        f"{tuple(dummy_vec.shape)}, costmap {tuple(dummy_costmap.shape)}, rnn_state {tuple(dummy_rnn_state.shape)}"
    )
    print(f"State normalization: {normalization}")


if __name__ == "__main__":
    main()

