# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject, RigidObjectCfg


class ObstacleManager:
    """Dynamic obstacles: positions/velocities, movement with boundary bounce, collision mask, reset."""

    def __init__(self, cfg, device: torch.device | str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.obstacle_initial_pos = [
            torch.zeros((num_envs, 3), device=device) for _ in range(cfg.num_obstacles)
        ]
        self.obstacle_velocity = [
            torch.zeros((num_envs, 3), device=device) for _ in range(cfg.num_obstacles)
        ]
        self.obstacle_objects: list[RigidObject] = []

    def setup(self, scene) -> None:
        """Create RigidObject for each obstacle and register in scene. Call after clone_environments."""
        for i in range(self.cfg.num_obstacles):
            obs_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                spawn=self.cfg.obstacle_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
            )
            obj = RigidObject(cfg=obs_cfg)
            self.obstacle_objects.append(obj)
            if hasattr(scene, "rigid_objects"):
                scene.rigid_objects[f"obstacle_{i}"] = obj

    def move(self, dt: float, env_origins: torch.Tensor) -> None:
        """Advance obstacle positions (XY), bounce off boundary, write poses to PhysX."""
        R = self.cfg.obstacle_boundary_radius
        quat_wxyz = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        for i in range(self.cfg.num_obstacles):
            self.obstacle_initial_pos[i][:, 0] += self.obstacle_velocity[i][:, 0] * dt
            self.obstacle_initial_pos[i][:, 1] += self.obstacle_velocity[i][:, 1] * dt
            dist = torch.linalg.norm(self.obstacle_initial_pos[i][:, :2], dim=-1)
            over = dist > R
            if torch.any(over):
                xy = self.obstacle_initial_pos[i][:, :2]
                n = xy / (dist.unsqueeze(-1).clamp(min=1e-6))
                v = self.obstacle_velocity[i][:, :2]
                self.obstacle_velocity[i][:, :2] = torch.where(
                    over.unsqueeze(-1),
                    v - 2 * (v * n).sum(dim=-1, keepdim=True) * n,
                    v,
                )
                scale = R / dist.clamp(min=1e-6)
                self.obstacle_initial_pos[i][:, :2] = torch.where(
                    over.unsqueeze(-1),
                    xy * scale.unsqueeze(-1),
                    xy,
                )
            world_pos = env_origins.clone()
            world_pos[:, 0] += self.obstacle_initial_pos[i][:, 0]
            world_pos[:, 1] += self.obstacle_initial_pos[i][:, 1]
            world_pos[:, 2] = 0.25
            root_pose = torch.cat([world_pos, quat_wxyz], dim=-1)
            self.obstacle_objects[i].write_root_pose_to_sim(root_pose)
            self.obstacle_objects[i].write_root_velocity_to_sim(
                torch.zeros(self.num_envs, 6, device=self.device)
            )

    def collision_mask(
        self, robot_xy: torch.Tensor, env_origins: torch.Tensor
    ) -> torch.Tensor:
        """Return [num_envs, 1] bool: geometric collision (robot vs obstacle centers)."""
        collision_dist = self.cfg.collision_robot_radius + self.cfg.obstacle_cfg.radius
        has_collision = torch.zeros(
            (self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        for i in range(self.cfg.num_obstacles):
            obs_xy = env_origins[:, :2] + self.obstacle_initial_pos[i][:, :2]
            dist = torch.linalg.norm(robot_xy - obs_xy, dim=-1, keepdim=True)
            has_collision |= dist <= collision_dist
        return has_collision

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Randomize obstacle positions and velocities for env_ids, then push all poses to PhysX."""
        speed = self.cfg.obstacle_speed
        obs_angle_range = getattr(self.cfg, "obstacle_angle_range", None)
        n = len(env_ids)
        for i in range(self.cfg.num_obstacles):
            if obs_angle_range is not None:
                a0, a1 = obs_angle_range
                rand_angles = torch.rand(n, device=self.device) * (a1 - a0) + a0
            else:
                rand_angles = torch.rand(n, device=self.device) * 2 * math.pi
            r0, r1 = self.cfg.obstacle_radius_range
            rand_dists = torch.rand(n, device=self.device) * (r1 - r0) + r0
            self.obstacle_initial_pos[i][env_ids, 0] = rand_dists * torch.cos(rand_angles)
            self.obstacle_initial_pos[i][env_ids, 1] = rand_dists * torch.sin(rand_angles)
            self.obstacle_initial_pos[i][env_ids, 2] = 0.25
            vel_angles = torch.rand(n, device=self.device) * 2 * math.pi
            self.obstacle_velocity[i][env_ids, 0] = speed * torch.cos(vel_angles)
            self.obstacle_velocity[i][env_ids, 1] = speed * torch.sin(vel_angles)
            self.obstacle_velocity[i][env_ids, 2] = 0.0

        quat_wxyz = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        for i in range(self.cfg.num_obstacles):
            world_pos = env_origins.clone()
            world_pos[:, 0] += self.obstacle_initial_pos[i][:, 0]
            world_pos[:, 1] += self.obstacle_initial_pos[i][:, 1]
            world_pos[:, 2] = 0.25
            root_pose = torch.cat([world_pos, quat_wxyz], dim=-1)
            self.obstacle_objects[i].write_root_pose_to_sim(root_pose)
            self.obstacle_objects[i].write_root_velocity_to_sim(
                torch.zeros(self.num_envs, 6, device=self.device)
            )
