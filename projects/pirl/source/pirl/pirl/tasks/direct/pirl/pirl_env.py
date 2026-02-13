# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import MultiMeshRayCaster
import isaaclab.utils.math as math_utils
from isaaclab.sim import XformPrimView

from .pirl_env_cfg import PirlEnvCfg
from .pirl_env_costmap import LocalCostmapBuilder
from .pirl_env_path import LocalPathManager
from .pirl_env_visuals import define_markers, define_path_markers, visualize_markers


class PirlEnv(DirectRLEnv):
    cfg: PirlEnvCfg

    def __init__(self, cfg: PirlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        # Initialize buffers
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.yaws = torch.zeros((self.num_envs, 1), device=self.device)
        self.prev_target_dist = torch.zeros((self.num_envs, 1), device=self.device)
        self.prev_path_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.marker_offset = torch.tensor([0.0, 0.0, 0.5], device=self.device).repeat(self.num_envs, 1)
        
        # Local grid buffers (Nav2-like costmap)
        self.costmap = LocalCostmapBuilder(self.cfg, self.device, self.num_envs)

        # Local path buffers
        self.path_manager = LocalPathManager(self.cfg, self.device, self.num_envs)
        
        # Obstacle movement state
        self.obstacle_time = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_initial_pos = [torch.zeros((self.num_envs, 3), device=self.device) for _ in range(self.cfg.num_obstacles)]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane with explicit friction
        ground_cfg = GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.ground_static_friction,
                dynamic_friction=self.cfg.ground_dynamic_friction,
                friction_combine_mode=self.cfg.ground_friction_combine,
            )
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        
        # Add obstacles at initial spread positions to avoid "cage" at (0,0)
        for i in range(self.cfg.num_obstacles):
            angle = i * (2 * math.pi / self.cfg.num_obstacles)
            dist = 2.0
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            self.cfg.obstacle_cfg.func(
                f"/World/envs/env_.*/Obstacle_{i}", 
                self.cfg.obstacle_cfg, 
                translation=(x, y, 0.25)
            )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Initialize views for obstacles
        self.obstacle_views = []
        for i in range(self.cfg.num_obstacles):
            view = XformPrimView(f"/World/envs/env_.*/Obstacle_{i}")
            self.obstacle_views.append(view)
        
        # Initialize sensors
        self.lidar = MultiMeshRayCaster(self.cfg.lidar)

        self.scene.sensors["lidar"] = self.lidar
        
        # Initialize markers
        self.visualization_markers = define_markers()
        self.path_markers = define_path_markers()
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # PPO actions are sampled from an unbounded Gaussian; clamp to keep in [-1, 1]
        self.actions = torch.clamp(actions, -1.0, 1.0).clone()
        self._move_obstacles()
        self._visualize_markers()

    def _move_obstacles(self):
        """Move some obstacles in an oscillating pattern."""
        self.obstacle_time += self.cfg.sim.dt * self.cfg.decimation
        
        # Move obstacles 0 and 1
        for i in [0, 1]:
            offset_y = torch.sin(self.obstacle_time * 2.0) * 0.8
            
            current_pos = self.scene.env_origins.clone()
            current_pos[:, 0] += self.obstacle_initial_pos[i][:, 0]
            current_pos[:, 1] += self.obstacle_initial_pos[i][:, 1] + offset_y
            current_pos[:, 2] = 0.25
            
            self.obstacle_views[i].set_world_poses(current_pos, torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1))

    def _visualize_markers(self):
        visualize_markers(
            self.visualization_markers,
            self.path_markers,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.marker_offset,
            self.yaws,
            self.up_dir,
            self.path_manager.path_points_w,
            self.path_manager.path_idx,
            self.cfg,
            self.device,
        )

    def _apply_action(self) -> None:
        # Map normalized actions to cmd_vel, then to wheel angular speeds
        v = self.actions[:, 0] * self.cfg.max_lin_vel
        w = self.actions[:, 1] * self.cfg.max_ang_vel
        omega_r = (v + 0.5 * self.cfg.track_width * w) / self.cfg.wheel_radius
        omega_l = (v - 0.5 * self.cfg.track_width * w) / self.cfg.wheel_radius
        targets = torch.stack((omega_l, omega_r), dim=-1)
        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # Calculate forward vector in world frame
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_quat_w, 
            torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        )
        
        # Update commands from local path target
        robot_pos_w = self.robot.data.root_pos_w[:, :2]
        self.commands, self.yaws, curr_idx = self.path_manager.update_commands(robot_pos_w)
        # store current distance to target for progress reward
        curr_targets_w = self.path_manager.path_points_w[torch.arange(self.num_envs, device=self.device), curr_idx]
        to_target_w = curr_targets_w - robot_pos_w
        self.curr_target_dist = torch.linalg.norm(to_target_w, dim=-1, keepdim=True)

        # Dot product: alignment (-1 to 1)
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # Cross product (z-component): turn direction
        cross = (self.forwards[:, 0] * self.commands[:, 1] - self.forwards[:, 1] * self.commands[:, 0]).unsqueeze(-1)
        # Forward speed in body frame
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)

        # Lidar ranges normalized to [0, 1]
        # MultiMeshRayCaster stores hit positions; compute distances from ray starts
        ray_hits_w = self.lidar.data.ray_hits_w
        ray_starts_w = getattr(self.lidar, "_ray_starts_w", None)
        if ray_starts_w is None:
            raise RuntimeError("Lidar ray starts are not available for distance computation.")
        lidar_ranges = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
        lidar_ranges = torch.where(
            torch.isfinite(lidar_ranges),
            lidar_ranges,
            torch.tensor(self.cfg.lidar.max_distance, device=self.device),
        )
        lidar_ranges = torch.clamp(lidar_ranges, max=self.cfg.lidar.max_distance)
        lidar_ranges_m = lidar_ranges

        grid_obs = self.costmap.build_image(lidar_ranges_m)

        # Local path segment in robot frame
        path_obs = self.path_manager.get_segment(robot_pos_w, self.robot.data.root_quat_w, curr_idx)

        vec_obs = torch.hstack((dot, cross, forward_speed, path_obs))
        return {"policy": {"vec": vec_obs, "costmap": grid_obs}}

    def _get_rewards(self) -> torch.Tensor:
        progress = self.prev_target_dist - self.curr_target_dist
        self.prev_target_dist = self.curr_target_dist
        reward = progress * self.cfg.rew_scale_progress
        # bonus when a path goal point is reached (index advanced)
        reached_goal = self.path_manager.path_idx > self.prev_path_idx
        reward += reached_goal.unsqueeze(-1).float() * self.cfg.rew_goal_bonus
        self.prev_path_idx = self.path_manager.path_idx.clone()

        # penalize only actual obstacle contact (collision in XY)
        has_collision = self._compute_collision_mask()
        reward += has_collision.float() * self.cfg.rew_scale_collision
        # small time penalty to encourage faster completion
        reward += self.cfg.rew_step_penalty
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        path_done = self.path_manager.path_idx >= (self.cfg.path_num_points - 1)
        collision_done = self._compute_collision_mask().squeeze(-1)
        die = path_done | collision_done
        return die, time_out

    def _compute_collision_mask(self) -> torch.Tensor:
        """Return [num_envs, 1] bool collision mask using XY overlap with obstacle cylinders."""
        robot_pos_xy = self.robot.data.root_pos_w[:, :2]
        collision_distance = self.cfg.collision_robot_radius + self.cfg.obstacle_cfg.radius
        has_collision = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        for i in range(self.cfg.num_obstacles):
            obstacle_pos_xy = self.scene.env_origins[:, :2] + self.obstacle_initial_pos[i][:, :2]
            dist = torch.linalg.norm(robot_pos_xy - obstacle_pos_xy, dim=-1, keepdim=True)
            has_collision |= dist <= collision_distance
        return has_collision

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Randomize obstacle positions on reset
        for i in range(self.cfg.num_obstacles):
            rand_angles = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
            rand_dists = torch.rand(len(env_ids), device=self.device) * (self.cfg.obstacle_radius_range[1] - self.cfg.obstacle_radius_range[0]) + self.cfg.obstacle_radius_range[0]
            
            self.obstacle_initial_pos[i][env_ids, 0] = rand_dists * torch.cos(rand_angles)
            self.obstacle_initial_pos[i][env_ids, 1] = rand_dists * torch.sin(rand_angles)
            self.obstacle_initial_pos[i][env_ids, 2] = 0.25
            
            world_pos = self.scene.env_origins[env_ids].clone()
            world_pos[:, 0] += self.obstacle_initial_pos[i][env_ids, 0]
            world_pos[:, 1] += self.obstacle_initial_pos[i][env_ids, 1]
            world_pos[:, 2] = 0.25
            self.obstacle_views[i].set_world_poses(world_pos, torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(env_ids), 1), env_ids)

        self.obstacle_time[env_ids] = 0.0

        # Reset robot state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.06
        self.robot.write_root_state_to_sim(root_state, env_ids)
        
        # Reset sensors
        self.lidar.reset(env_ids)
        # Reset grid history
        self.costmap.reset(env_ids)
        # Reset path points
        env_origins = self.scene.env_origins[env_ids, :2]
        self.path_manager.reset(env_ids, env_origins)
        # Reset progress tracking
        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]
        curr_idx = self.path_manager.path_idx[env_ids]
        curr_targets_w = self.path_manager.path_points_w[env_ids, curr_idx]
        to_target_w = curr_targets_w - robot_pos_w
        self.prev_target_dist[env_ids] = torch.linalg.norm(to_target_w, dim=-1, keepdim=True)
        self.prev_path_idx[env_ids] = self.path_manager.path_idx[env_ids]