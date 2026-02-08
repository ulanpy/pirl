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
from isaaclab.sensors import ContactSensor, MultiMeshRayCaster
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.sim import XformPrimView

from .pirl_env_cfg import PirlEnvCfg


def define_markers(num_envs, device) -> VisualizationMarkers:
    """Define markers for robot orientation and command direction."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class PirlEnv(DirectRLEnv):
    cfg: PirlEnvCfg

    def __init__(self, cfg: PirlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        # Initialize buffers
        self.has_collision = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.yaws = torch.zeros((self.num_envs, 1), device=self.device)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.marker_offset = torch.tensor([0.0, 0.0, 0.5], device=self.device).repeat(self.num_envs, 1)
        
        # Obstacle movement state
        self.obstacle_time = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_initial_pos = [torch.zeros((self.num_envs, 3), device=self.device) for _ in range(self.cfg.num_obstacles)]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
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
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.lidar = MultiMeshRayCaster(self.cfg.lidar)
        
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["lidar"] = self.lidar
        
        # Initialize markers
        self.visualization_markers = define_markers(self.num_envs, self.device)
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
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
        # Get marker locations and orientations
        marker_locations = self.robot.data.root_pos_w + self.marker_offset
        forward_orientations = self.robot.data.root_quat_w
        command_orientations = math_utils.quat_from_angle_axis(self.yaws.squeeze(-1), self.up_dir)

        # Stack for visualization
        locs = torch.vstack((marker_locations, marker_locations))
        rots = torch.vstack((forward_orientations, command_orientations))
        
        # Indices: 0 for forward, 1 for command
        indices = torch.hstack((
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long), 
            torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        ))
        self.visualization_markers.visualize(locs, rots, marker_indices=indices)

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions * self.cfg.action_scale, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # Calculate forward vector in world frame
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_quat_w, 
            torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        )
        
        # Dot product: alignment (-1 to 1)
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # Cross product (z-component): turn direction
        cross = (self.forwards[:, 0] * self.commands[:, 1] - self.forwards[:, 1] * self.commands[:, 0]).unsqueeze(-1)
        # Forward speed in body frame
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)
        
        obs = torch.hstack((dot, cross, forward_speed))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)
        alignment = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        
        # Reward = Speed * exp(Alignment) - Penalty
        reward = forward_speed * torch.exp(alignment)
        reward += self.has_collision.float() * self.cfg.rew_scale_collision
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for collisions with a higher threshold
        collision_forces = torch.linalg.norm(self.contact_sensor.data.net_forces_w_history, dim=-1)
        self.has_collision = torch.any(collision_forces > 300.0, dim=(1, 2)).unsqueeze(-1)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        die = self.has_collision.squeeze(-1)
        return die, time_out

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

        # New random commands
        new_cmds = torch.randn((len(env_ids), 3), device=self.device)
        new_cmds[:, 2] = 0.0
        self.commands[env_ids] = new_cmds / torch.linalg.norm(new_cmds, dim=1, keepdim=True)
        self.yaws[env_ids] = torch.atan2(self.commands[env_ids, 1], self.commands[env_ids, 0]).unsqueeze(-1)
        self.obstacle_time[env_ids] = 0.0

        # Reset robot state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.06
        self.robot.write_root_state_to_sim(root_state, env_ids)
        
        # Reset sensors
        self.contact_sensor.reset(env_ids)
        self.lidar.reset(env_ids)
