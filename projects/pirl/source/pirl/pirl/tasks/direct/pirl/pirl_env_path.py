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

    def _generate_path_for_env(self, origin_xy: torch.Tensor) -> torch.Tensor:
        """Step-by-step curved path generation without static obstacle constraints."""
        k = self.cfg.path_num_points
        r_min, r_max = self.cfg.path_radius_range
        noise_scale = float(getattr(self.cfg, "path_heading_noise_scale", 0.35))
        mid_turn = float(getattr(self.cfg, "path_mid_turn_rad", 0.5))

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

            heading_try = base_heading + float((torch.rand(1, device=self.device).item() - 0.5) * 2.0 * noise_scale)
            direction = torch.tensor([math.cos(heading_try), math.sin(heading_try)], device=self.device)
            candidate = curr + step_len * direction
            path[i] = candidate
            curr = candidate
            prev_radius = float(torch.linalg.norm(candidate - origin_xy).item())
            heading = heading_try
        return path

    def reset(
        self,
        env_ids,
        env_origins: torch.Tensor,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, (list, range)):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        n = env_ids.shape[0]
        env_origins_sub = self._extract_env_origins_xy(env_ids, env_origins)

        path_points = torch.zeros((n, self.cfg.path_num_points, 2), device=self.device)
        for i in range(n):
            path_points[i] = self._generate_path_for_env(env_origins_sub[i])

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
