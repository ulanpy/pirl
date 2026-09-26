from __future__ import annotations

import torch
from typing import Any, cast

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] | None = None  # populated after class definition


class DifferentialDriveAction(ActionTerm):
    """Normalized linear/yaw command mapped to Burger wheel velocity targets."""

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def __init__(self, cfg: DifferentialDriveActionCfg, env) -> None:
        super().__init__(cfg, env)
        asset = cast(Articulation, self._asset)
        self._env = cast(Any, env)
        self._dof_idx, _ = asset.find_joints(self._env.task_cfg.dof_names)
        self._raw_actions = torch.zeros((env.num_envs, 2), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.prev_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self._processed_actions)
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(torch.clamp(actions, -1.0, 1.0))

    def apply_actions(self) -> None:
        cfg = self._env.task_cfg
        v = self._processed_actions[:, 0] * float(cfg.max_lin_vel)
        w = self._processed_actions[:, 1] * float(cfg.max_ang_vel)
        targets = torch.stack(
            ((v - 0.5 * float(cfg.track_width) * w) / float(cfg.wheel_radius),
             (v + 0.5 * float(cfg.track_width) * w) / float(cfg.wheel_radius)),
            dim=-1,
        )
        cast(Articulation, self._asset).set_joint_velocity_target(targets, joint_ids=self._dof_idx)
        if self._env.dyn_obstacles is not None:
            self._env.dyn_obstacles.step(self._env.physics_dt, self._env.scene.env_origins)


DifferentialDriveActionCfg.class_type = DifferentialDriveAction
