from collections.abc import Sequence
import math
import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg
import isaaclab.utils.math as math_utils

from .pirl_env_cfg import PirlEnvCfg
from .pirl_env_costmap import LocalCostmapBuilder
from .pirl_env_dyn_obstacles import DynamicObstaclesManager
from .pirl_env_dr_obstacles import DomainRandomizationObstacles
from .pirl_env_path import LocalPathManager
from .pirl_env_proximity import ProximityReward
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
        self._latest_lidar_ranges_m = None
        self.proximity = ProximityReward(self.cfg, self.device, self.num_envs)
        self.extras = {}
        # For action-rate reward: current and previous step actions (shape [num_envs, 2])
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 2), device=self.device)
        # For proximity closing-rate reward: previous min distance in selected proximity rays
        self.prev_min_prox_range = torch.full(
            (self.num_envs, 1), float(self.cfg.lidar.max_distance), device=self.device
        )
        # IRA setup deferred to first reset() so NavMesh bakes after sim has run and scene is composed.

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # Spawn pre-generated SceneBlox USD scene into env_0 before cloning.
        # With replicate_physics=True this propagates the same static map to all envs.
        scene_paths = tuple(getattr(self.cfg, "sceneblox_usd_paths", ()))
        if len(scene_paths) == 0:
            raise RuntimeError("sceneblox_usd_paths is empty.")
        scene_usd = scene_paths[0]
        is_remote_usd = "://" in scene_usd
        if (not is_remote_usd) and (not os.path.exists(scene_usd)):
            raise FileNotFoundError(
                f"Scene USD not found: {scene_usd}. Generate SceneBlox USDs first."
            )
        scene_cfg = sim_utils.UsdFileCfg(usd_path=scene_usd)
        scene_cfg.func(
            "/World/envs/env_0/GeneratedScene",
            scene_cfg,
            translation=(0.0, 0.0, 0.0),
        )

        # copy_from_source=True: each env gets full copy so dr_obstacles can add Obstacle_* per env.
        self.scene.clone_environments(copy_from_source=True)
        self.scene.articulations["robot"] = self.robot
        self.dr_obstacles = DomainRandomizationObstacles(
            self.cfg, self.device, self.num_envs
        )
        self.dr_obstacles.setup()
        self.dyn_obstacles = DynamicObstaclesManager(self.cfg, self.device, self.num_envs)
        self.dyn_obstacles.setup()
        self.cfg.lidar.mesh_prim_paths = self._build_static_lidar_targets_from_stage()
        # Dynamic obstacle meshes as lidar targets.
        if self.cfg.dyn_obstacle_enabled:
            self.cfg.lidar.mesh_prim_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/envs/env_.*/GeneratedScene/DynamicObstacles/DynObstacle_.*",
                    track_mesh_transforms=True,
                )
            )

        # Initialize sensors
        self.lidar = MultiMeshRayCaster(self.cfg.lidar)
        if self.cfg.lidar.debug_vis:
            self._patch_lidar_debug_vis_callback()
        self.scene.sensors["lidar"] = self.lidar

        # Initialize markers
        self.visualization_markers = define_markers()
        self.path_markers = define_path_markers()

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_static_lidar_targets_from_stage(self) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
        """Build exact static lidar target expressions from loaded scene meshes."""
        fallback_targets = list(self.cfg.lidar.mesh_prim_paths)
        stage = sim_utils.get_current_stage()
        root_path = "/World/envs/env_0/GeneratedScene"
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return fallback_targets

        top_level_with_mesh: set[str] = set()
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            prim_path = str(prim.GetPrimPath())
            if prim.GetTypeName() == "Mesh" and "/collisions/" not in prim_path:
                rel = prim_path[len(root_path) + 1 :]
                if rel:
                    top = rel.split("/", 1)[0]
                    if top != "DynamicObstacles":
                        top_level_with_mesh.add(top)
            for child in prim.GetChildren():
                stack.append(child)

        if not top_level_with_mesh:
            return fallback_targets

        targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
        for top in sorted(top_level_with_mesh):
            targets.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=f"/World/envs/env_.*/GeneratedScene/{top}.*",
                    track_mesh_transforms=False,
                )
            )
        return targets

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Save previous action before updating (for action-rate penalty in reward)
        self.prev_actions.copy_(self.actions)
        # PPO actions are sampled from an unbounded Gaussian; clamp to keep in [-1, 1]
        self.actions = torch.clamp(actions, -1.0, 1.0).clone()
        # Update moving obstacle poses each env step.
        self.dyn_obstacles.step(self.cfg.sim.dt * self.cfg.decimation, self.scene.env_origins)
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

    def _patch_lidar_debug_vis_callback(self) -> None:
        """Render stable lidar debug points (hits + max-range misses) without callback spam."""

        def _safe_debug_callback(event):
            # During startup callback can fire before ray-caster fully initializes internal buffers.
            if (
                not hasattr(self.lidar, "ray_visualizer")
                or not hasattr(self.lidar, "drift")
                or not hasattr(self.lidar, "_data")
            ):
                return
            ray_hits_w = self.lidar._data.ray_hits_w
            if ray_hits_w is None:
                return
            ray_starts_w = getattr(self.lidar, "_ray_starts_w", None)
            ray_dirs_w = getattr(self.lidar, "_ray_directions_w", None)
            if ray_starts_w is None or ray_dirs_w is None:
                return
            # Build visualization points: true hits; misses shown at max range endpoint.
            miss_mask = torch.any(~torch.isfinite(ray_hits_w), dim=2)
            viz_points = ray_hits_w.clone()
            viz_points[miss_mask] = (
                ray_starts_w[miss_mask] + ray_dirs_w[miss_mask] * float(self.cfg.lidar.max_distance)
            )
            viz_points = viz_points.reshape(-1, 3)
            if viz_points.numel() == 0:
                return
            self.lidar.ray_visualizer.visualize(viz_points)

        self.lidar._debug_vis_callback = _safe_debug_callback

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
        self.dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # Cross product (z-component): turn direction
        cross = (self.forwards[:, 0] * self.commands[:, 1] - self.forwards[:, 1] * self.commands[:, 0]).unsqueeze(-1)
        # Ego-motion in body frame
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)
        lateral_speed = self.robot.data.root_com_lin_vel_b[:, 1].unsqueeze(-1)
        yaw_rate = self.robot.data.root_com_ang_vel_b[:, 2].unsqueeze(-1)

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
        self._latest_lidar_ranges_m = lidar_ranges_m

        robot_yaw = torch.atan2(self.forwards[:, 1], self.forwards[:, 0])
        grid_obs = self.costmap.build_image(lidar_ranges_m, robot_pos_w, robot_yaw)

        # Local path segment in robot frame
        path_obs = self.path_manager.get_segment(robot_pos_w, self.robot.data.root_quat_w, curr_idx)

        vec_obs = torch.hstack((self.dot, cross, forward_speed, lateral_speed, yaw_rate, path_obs))
        return {"policy": {"vec": vec_obs, "costmap": grid_obs}}

    def _get_rewards(self) -> torch.Tensor:
            # 1. Вычисляем отдельные компоненты
            # Прогресс к цели (только поощряем приближение; отдаление/смена waypoint не штрафуем)
            progress_val = torch.clamp(
                self.prev_target_dist - self.curr_target_dist, min=0.0
            ) * self.cfg.rew_scale_progress
            
            # Направление движения (keep shape [num_envs, 1] for reward sum and memory)
            heading_val = (self.dot * self.cfg.rew_scale_heading)

            # Штраф за скорость изменения yaw-команды [num_envs, 1]
            yaw_delta = self.actions[:, 1:2] - self.prev_actions[:, 1:2]
            rew_action_rate = self.cfg.rew_scale_action_rate * torch.square(yaw_delta)
            
            # Штраф за близость (proximity)
            proximity_val = self.proximity.compute_penalty(self._latest_lidar_ranges_m)
            # Штраф за скорость сближения с препятствием (dd/dt), активен только в ближней зоне.
            if self._latest_lidar_ranges_m is None:
                rew_proximity_rate = torch.zeros((self.num_envs, 1), device=self.device)
            else:
                curr_min_prox_range = self.proximity.min_selected_range(self._latest_lidar_ranges_m)
                dt_env = self.cfg.sim.dt * self.cfg.decimation
                closing_speed = torch.clamp(
                    -(curr_min_prox_range - self.prev_min_prox_range) / dt_env, min=0.0
                )
                gate = curr_min_prox_range < float(self.cfg.proximity_rate_gate_distance)
                rew_proximity_rate = (
                    float(self.cfg.rew_scale_proximity_rate) * closing_speed * gate.float()
                )
            
            # Бонус за достижение точки пути
            reached_goal = self.path_manager.path_idx > self.prev_path_idx
            goal_bonus_val = reached_goal.unsqueeze(dim=-1).float() * self.cfg.rew_goal_bonus
            
            # Collision reward (no obstacles in empty scene → always 0)
            collision_val = torch.zeros((self.num_envs, 1), device=self.device)

            # Скорость и реверс
            forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)
            reverse_val = torch.zeros_like(forward_speed)
            if self.cfg.rew_scale_reverse != 0:
                reverse_val = float(self.cfg.rew_scale_reverse) * torch.clamp(-forward_speed, min=0.0)
                
            # Постоянный штраф за время
            step_penalty_val = torch.full_like(forward_speed, self.cfg.rew_step_penalty)

            # 2. Итоговая награда
            reward = (
                progress_val + 
                proximity_val + 
                goal_bonus_val + 
                collision_val + 
                reverse_val + 
                step_penalty_val + 
                heading_val + 
                rew_action_rate +
                rew_proximity_rate
            )

            # 3. Обновляем буферы состояния
            self.prev_target_dist = self.curr_target_dist
            self.prev_path_idx = self.path_manager.path_idx.clone()
            if self._latest_lidar_ranges_m is not None:
                self.prev_min_prox_range = curr_min_prox_range

            # 4. Reward Breakdown для логирования (TensorBoard/WandB)
            # Мы используем .mean(), чтобы получить среднее значение по всем параллельным средам
            if "log" not in self.extras:
                self.extras["log"] = {}
            
            self.extras["log"]["rew/progress"] = torch.mean(progress_val)
            self.extras["log"]["rew/proximity"] = torch.mean(proximity_val)
            self.extras["log"]["rew/goal_bonus"] = torch.mean(goal_bonus_val)
            self.extras["log"]["rew/collision"] = torch.mean(collision_val)
            self.extras["log"]["rew/reverse"] = torch.mean(reverse_val)
            self.extras["log"]["rew/step_penalty"] = torch.mean(step_penalty_val)
            self.extras["log"]["rew/heading"] = torch.mean(heading_val)
            self.extras["log"]["rew/action_rate"] = torch.mean(rew_action_rate)
            self.extras["log"]["rew/proximity_rate"] = torch.mean(rew_proximity_rate)
            self.extras["log"]["rew/total"] = torch.mean(reward)

            return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # path_done = self.path_manager.path_idx >= 1
        collision_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #die = path_done | collision_done
        die = collision_done
        return die, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_seq = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids_seq = env_ids.tolist()
        else:
            env_ids_seq = list(env_ids)
        env_ids_t = torch.as_tensor(env_ids_seq, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids_seq)

        # Reset robot state: random XY in disk of robot_spawn_radius, optionally in a "start zone" sector
        root_state = self.robot.data.default_root_state[env_ids_t].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids_t]
        spawn_r = self.cfg.robot_spawn_radius * torch.sqrt(torch.rand(len(env_ids_t), device=self.device))
        spawn_angle_range = getattr(self.cfg, "spawn_angle_range", None)
        if spawn_angle_range is not None:
            a0, a1 = spawn_angle_range
            spawn_theta = torch.rand(len(env_ids_t), device=self.device) * (a1 - a0) + a0
        else:
            spawn_theta = torch.rand(len(env_ids_t), device=self.device) * 2 * math.pi
        root_state[:, 0] += spawn_r * torch.cos(spawn_theta)
        root_state[:, 1] += spawn_r * torch.sin(spawn_theta)
        # When using a start zone, face robot toward arena (origin) so path is ahead
        if spawn_angle_range is not None:
            yaw = spawn_theta + math.pi
            half = 0.5 * yaw
            root_state[:, 3] = torch.cos(half)
            root_state[:, 4] = 0.0
            root_state[:, 5] = 0.0
            root_state[:, 6] = torch.sin(half)
        # PhysX backend expects tensor-like indices here.
        self.robot.write_root_state_to_sim(root_state, env_ids_t)  # type: ignore[arg-type]

        # Episode-level domain randomization for static obstacles.
        self.dr_obstacles.reset(env_ids_t, self.scene.env_origins)
        # Runtime dynamic obstacles.
        self.dyn_obstacles.reset(env_ids_t, self.scene.env_origins)
        
        # Reset sensors
        self.lidar.reset(env_ids_seq)
        # Reset grid history
        self.costmap.reset(env_ids_seq)
        # Reset path points
        env_origins = self.scene.env_origins[env_ids_t, :2]
        self.path_manager.reset(env_ids_seq, env_origins)
        # Reset progress tracking
        robot_pos_w = self.robot.data.root_pos_w[env_ids_t, :2]
        curr_idx = self.path_manager.path_idx[env_ids_t]
        curr_targets_w = self.path_manager.path_points_w[env_ids_t, curr_idx]
        to_target_w = curr_targets_w - robot_pos_w
        self.prev_target_dist[env_ids_t] = torch.linalg.norm(to_target_w, dim=-1, keepdim=True)
        self.prev_path_idx[env_ids_t] = self.path_manager.path_idx[env_ids_t]
        self.prev_min_prox_range[env_ids_t] = float(self.cfg.lidar.max_distance)