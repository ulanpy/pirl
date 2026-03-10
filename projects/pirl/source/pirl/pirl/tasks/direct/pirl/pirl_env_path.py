import math

import torch
import isaaclab.utils.math as math_utils


class LocalPathManager:
    def __init__(self, cfg, device: str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.path_points_w = torch.zeros(
            (num_envs, cfg.path_num_points, 2), device=device
        )
        self.path_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Fallback when robot is on top of waypoint (to_target ~ 0) so command stays non-zero
        self._last_command_w = torch.zeros((num_envs, 3), device=device)
        self._last_command_w[:, 0] = 1.0

    def _extract_env_origins_xy(
        self, env_ids: torch.Tensor, env_origins: torch.Tensor
    ) -> torch.Tensor:
        """Accept either all-env origins or pre-sliced origins, always return (n, 2)."""
        n = env_ids.shape[0]
        if env_origins.shape[0] == self.num_envs:
            return env_origins[env_ids].reshape(n, -1)[:, :2]
        if env_origins.shape[0] == n:
            return env_origins.reshape(n, -1)[:, :2]
        raise ValueError(
            f"Unexpected env_origins shape {tuple(env_origins.shape)} for n={n}, num_envs={self.num_envs}."
        )

    def _points_collide_obbs(
        self, points_xy: torch.Tensor, obbs: torch.Tensor, margin: float
    ) -> bool:
        """Check if any point lies inside any inflated OBB (x, y, yaw, hx, hy)."""
        if obbs.shape[0] == 0:
            return False
        rel = points_xy.unsqueeze(1) - obbs[:, :2].unsqueeze(0)  # (P, M, 2)
        cos_yaw = torch.cos(obbs[:, 2]).unsqueeze(0)  # (1, M)
        sin_yaw = torch.sin(obbs[:, 2]).unsqueeze(0)  # (1, M)
        local_x = rel[..., 0] * cos_yaw + rel[..., 1] * sin_yaw
        local_y = -rel[..., 0] * sin_yaw + rel[..., 1] * cos_yaw
        hx = obbs[:, 3].unsqueeze(0) + margin
        hy = obbs[:, 4].unsqueeze(0) + margin
        inside = (torch.abs(local_x) <= hx) & (torch.abs(local_y) <= hy)
        return bool(inside.any().item())

    def _segment_collides_obbs(
        self, p0: torch.Tensor, p1: torch.Tensor, obbs: torch.Tensor, margin: float, step: float
    ) -> bool:
        """Sample segment and test sampled points against inflated OBBs."""
        if obbs.shape[0] == 0:
            return False
        seg = p1 - p0
        length = float(torch.linalg.norm(seg).item())
        n_samples = max(2, int(math.ceil(length / max(step, 1e-3))) + 1)
        ts = torch.linspace(0.0, 1.0, n_samples, device=self.device).unsqueeze(1)
        points = p0.unsqueeze(0) + ts * seg.unsqueeze(0)
        return self._points_collide_obbs(points, obbs, margin)

    def _generate_path_for_env(self, origin_xy: torch.Tensor, obbs: torch.Tensor) -> torch.Tensor:
        """Step-by-step path generation with OBB-aware rejection and fallback."""
        k = self.cfg.path_num_points
        r_min, r_max = self.cfg.path_radius_range
        noise_scale = float(getattr(self.cfg, "path_heading_noise_scale", 0.35))
        mid_turn = float(getattr(self.cfg, "path_mid_turn_rad", 0.5))
        obb_margin = float(getattr(self.cfg, "path_obb_margin", 0.25))
        seg_step = float(getattr(self.cfg, "path_segment_check_step", 0.12))
        max_step_resample = int(getattr(self.cfg, "path_step_max_resample", 20))

        angle_range = getattr(self.cfg, "path_angle_range", None)
        if angle_range is not None:
            a0, a1 = angle_range
            heading = float((torch.rand(1, device=self.device) * (a1 - a0) + a0).item())
        else:
            heading = float((torch.rand(1, device=self.device) * 2.0 * math.pi).item())

        path = torch.zeros((k, 2), device=self.device)
        radial_targets = torch.linspace(r_min, r_max, k, device=self.device)
        prev_radius = 0.0
        curr = origin_xy.clone()
        turn_applied = False

        for i in range(k):
            target_radius = float(radial_targets[i].item())
            step_len = max(0.2, target_radius - prev_radius)
            base_heading = heading
            if (i >= (k // 2)) and (not turn_applied):
                base_heading += float((torch.rand(1, device=self.device).item() - 0.5) * 2.0 * mid_turn)
                turn_applied = True

            accepted = False
            for _ in range(max_step_resample):
                heading_try = base_heading + float((torch.rand(1, device=self.device).item() - 0.5) * 2.0 * noise_scale)
                direction = torch.tensor([math.cos(heading_try), math.sin(heading_try)], device=self.device)
                candidate = curr + step_len * direction
                candidate_radius = float(torch.linalg.norm(candidate - origin_xy).item())
                if candidate_radius < prev_radius - 0.03:
                    continue
                if self._segment_collides_obbs(curr, candidate, obbs, obb_margin, seg_step):
                    continue
                path[i] = candidate
                curr = candidate
                prev_radius = candidate_radius
                heading = heading_try
                accepted = True
                break

            if not accepted:
                # Deterministic fallback sweep around current heading for hard scenes.
                fallback_done = False
                for delta in (0.45, -0.45, 0.9, -0.9, 1.35, -1.35):
                    heading_try = base_heading + delta
                    direction = torch.tensor([math.cos(heading_try), math.sin(heading_try)], device=self.device)
                    candidate = curr + step_len * direction
                    if self._segment_collides_obbs(curr, candidate, obbs, obb_margin, seg_step):
                        continue
                    path[i] = candidate
                    curr = candidate
                    prev_radius = float(torch.linalg.norm(candidate - origin_xy).item())
                    heading = heading_try
                    fallback_done = True
                    break
                if not fallback_done:
                    # Last-resort: keep moving slightly forward to preserve continuity.
                    candidate = curr + 0.25 * torch.tensor([math.cos(base_heading), math.sin(base_heading)], device=self.device)
                    path[i] = candidate
                    curr = candidate
                    prev_radius = float(torch.linalg.norm(candidate - origin_xy).item())
                    heading = base_heading
        return path

    def reset(
        self,
        env_ids,
        env_origins: torch.Tensor,
        obstacle_obbs_per_env: list[torch.Tensor] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, (list, range)):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        n = env_ids.shape[0]
        max_resample = int(getattr(self.cfg, "path_obstacle_max_resample", 12))
        env_origins_sub = self._extract_env_origins_xy(env_ids, env_origins)

        if obstacle_obbs_per_env is None:
            obstacles = [torch.zeros(0, 5, device=self.device) for _ in range(n)]
        else:
            if len(obstacle_obbs_per_env) != n:
                raise ValueError(
                    f"obstacle_obbs_per_env length {len(obstacle_obbs_per_env)} "
                    f"must match number of reset envs {n}."
                )
            obstacles = obstacle_obbs_per_env

        path_points = torch.zeros((n, self.cfg.path_num_points, 2), device=self.device)
        obb_margin = float(getattr(self.cfg, "path_obb_margin", 0.25))
        seg_step = float(getattr(self.cfg, "path_segment_check_step", 0.12))
        for i in range(n):
            path = self._generate_path_for_env(env_origins_sub[i], obstacles[i])
            tries = 0
            while tries < max_resample:
                has_collision = False
                for j in range(path.shape[0]):
                    p0 = env_origins_sub[i] if j == 0 else path[j - 1]
                    p1 = path[j]
                    if self._segment_collides_obbs(p0, p1, obstacles[i], obb_margin, seg_step):
                        has_collision = True
                        break
                if not has_collision:
                    break
                path = self._generate_path_for_env(env_origins_sub[i], obstacles[i])
                tries += 1
            path_points[i] = path

        self.path_points_w[env_ids] = path_points
        self.path_idx[env_ids] = 0
        self._last_command_w[env_ids, 0] = 1.0
        self._last_command_w[env_ids, 1] = 0.0
        self._last_command_w[env_ids, 2] = 0.0

    def update_commands(self, robot_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_idx = torch.clamp(self.path_idx, max=self.cfg.path_num_points - 1)
        curr_targets_w = self.path_points_w[torch.arange(self.num_envs, device=self.device), curr_idx]
        to_target_w = curr_targets_w - robot_pos_w
        dist_to_target = torch.linalg.norm(to_target_w, dim=-1, keepdim=True)
        advance = dist_to_target.squeeze(-1) < self.cfg.path_goal_threshold
        self.path_idx = torch.where(
            advance,
            torch.clamp(self.path_idx + 1, max=self.cfg.path_num_points - 1),
            self.path_idx,
        )
        curr_idx = torch.clamp(self.path_idx, max=self.cfg.path_num_points - 1)
        curr_targets_w = self.path_points_w[torch.arange(self.num_envs, device=self.device), curr_idx]
        to_target_w = curr_targets_w - robot_pos_w
        to_target_w_3 = torch.zeros((self.num_envs, 3), device=self.device)
        to_target_w_3[:, :2] = to_target_w
        to_target_norm = torch.linalg.norm(to_target_w_3, dim=-1, keepdim=True)
        # Use last non-zero command when on top of waypoint so every env has a valid direction
        small = (to_target_norm.squeeze(-1) < 1e-3)
        commands = torch.where(
            small.unsqueeze(-1),
            self._last_command_w,
            to_target_w_3 / to_target_norm.clamp(min=1e-6),
        )
        self._last_command_w.copy_(commands)
        yaws = torch.atan2(commands[:, 1], commands[:, 0]).unsqueeze(-1)
        return commands, yaws, curr_idx

    def get_segment(self, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor, curr_idx: torch.Tensor) -> torch.Tensor:
        seg_len = self.cfg.path_segment_len
        seg_indices = curr_idx.unsqueeze(1) + torch.arange(seg_len, device=self.device).unsqueeze(0)
        seg_indices = torch.clamp(seg_indices, max=self.cfg.path_num_points - 1)
        path_seg_w = self.path_points_w[torch.arange(self.num_envs, device=self.device).unsqueeze(1), seg_indices]
        rel_seg_w = path_seg_w - robot_pos_w.unsqueeze(1)
        rel_seg_w_3 = torch.zeros((self.num_envs, seg_len, 3), device=self.device)
        rel_seg_w_3[:, :, :2] = rel_seg_w
        rel_seg_b = math_utils.quat_apply_inverse(
            robot_quat_w.repeat_interleave(seg_len, dim=0), rel_seg_w_3.reshape(-1, 3)
        ).reshape(self.num_envs, seg_len, 3)[:, :, :2]
        return rel_seg_b.reshape(self.num_envs, -1)
