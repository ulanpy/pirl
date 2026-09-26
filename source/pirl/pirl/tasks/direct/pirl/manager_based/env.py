from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

from ..pirl_env_cfg import PirlTaskCfg
from ..pirl_env_costmap import LocalCostmapBuilder
from ..pirl_env_dyn_obstacles import DynamicObstacles
from ..pirl_env_path import LocalPathManager
from . import visuals


class PirlManagerEnv(ManagerBasedRLEnv):
    """Manager-based PIRL environment with one shared navigation-state refresh."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self.task_cfg = PirlTaskCfg()
        self._state_step = -1
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self) -> None:
        self.robot = self.scene["robot"]
        self.lidar = self.scene["lidar"]
        self.dyn_obstacles = DynamicObstacles(self.task_cfg, self.device, self.num_envs)
        self.dyn_obstacles.bind(self.scene["dyn_obstacles"])
        self.costmap = LocalCostmapBuilder(self.task_cfg, self.device, self.num_envs)
        self.path_manager = LocalPathManager(self.task_cfg, self.device, self.num_envs)
        self.heading_markers = visuals.create_heading_markers()
        self.path_markers = visuals.create_path_markers()
        self.prev_path_s = torch.zeros((self.num_envs, 1), device=self.device)
        self.curr_path_s = torch.zeros_like(self.prev_path_s)
        self.curr_path_error = torch.zeros_like(self.prev_path_s)
        self.curr_path_error_signed = torch.zeros_like(self.prev_path_s)
        self.path_heading_cos = torch.zeros_like(self.prev_path_s)
        self.final_goal_dist = torch.zeros_like(self.prev_path_s)
        self.prev_reward_components = torch.zeros(
            (self.num_envs, int(self.task_cfg.reward_component_dim)), device=self.device
        )
        self.reward_components = torch.zeros_like(self.prev_reward_components)
        self._latest_lidar_ranges_m = None
        super().load_managers()

    def refresh_navigation_state(self) -> None:
        if self._state_step == self._sim_step_counter:
            return
        robot_pos = self.robot.data.root_pos_w[:, :2]
        forwards = math_utils.quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        (_, _, _, self.curr_path_s, self.curr_path_error, self.curr_path_error_signed, _, heading_target) = (
            self.path_manager.update_commands(robot_pos)
        )
        yaw = torch.atan2(forwards[:, 1], forwards[:, 0])
        self.heading_error = torch.atan2(
            torch.sin(yaw.unsqueeze(-1) - heading_target), torch.cos(yaw.unsqueeze(-1) - heading_target)
        )
        self.path_heading_cos = torch.cos(self.heading_error)
        visuals.update(
            self.heading_markers,
            self.path_markers,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            heading_target,
            self.path_manager.path_points_w,
            self.path_manager.path_idx,
            int(self.task_cfg.path_segment_len),
        )
        self.final_goal_dist = torch.linalg.norm(self.path_manager.path_points_w[:, -1] - robot_pos, dim=-1, keepdim=True)
        hits = self.lidar.data.ray_hits_w
        starts = self.lidar._ray_starts_w
        ranges = torch.linalg.norm(hits - starts, dim=-1)
        self._latest_lidar_ranges_m = torch.where(
            torch.isfinite(ranges), ranges, torch.full_like(ranges, float(self.task_cfg.lidar.max_distance))
        ).clamp(max=float(self.task_cfg.lidar.max_distance))
        self.current_costmap = self.costmap.build_image(self._latest_lidar_ranges_m, robot_pos, yaw)
        self._state_step = self._sim_step_counter

    def invalidate_navigation_state(self) -> None:
        self._state_step = -1
