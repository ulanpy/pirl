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
        # Cumulative arc-length coordinate s for each waypoint.
        self.path_s = torch.zeros((num_envs, cfg.path_num_points), device=device)
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
        step_len = float(getattr(self.cfg, "path_point_spacing_m", 0.05))
        noise_scale = float(getattr(self.cfg, "path_heading_noise_scale", 0.35))
        mid_turn = float(getattr(self.cfg, "path_mid_turn_rad", 0.5))

        angle_range = getattr(self.cfg, "path_angle_range", None)
        if angle_range is not None:
            a0, a1 = angle_range
            heading = float((torch.rand(1, device=self.device) * (a1 - a0) + a0).item())
        else:
            heading = float((torch.rand(1, device=self.device) * 2.0 * math.pi).item())

        path = torch.zeros((k, 2), device=self.device)
        curr = origin_xy.clone()
        turn_applied = False

        for i in range(k):
            base_heading = heading
            if (i >= (k // 2)) and (not turn_applied):
                base_heading += float((torch.rand(1, device=self.device).item() - 0.5) * 2.0 * mid_turn)
                turn_applied = True

            heading_try = base_heading + float((torch.rand(1, device=self.device).item() - 0.5) * 2.0 * noise_scale)
            direction = torch.tensor([math.cos(heading_try), math.sin(heading_try)], device=self.device)
            candidate = curr + step_len * direction
            path[i] = candidate
            curr = candidate
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
        seg = path_points[:, 1:, :] - path_points[:, :-1, :]
        seg_lens = torch.linalg.norm(seg, dim=-1)
        path_s = torch.zeros((n, self.cfg.path_num_points), device=self.device)
        path_s[:, 1:] = torch.cumsum(seg_lens, dim=1)
        self.path_s[env_ids] = path_s
        self.path_idx[env_ids] = 0
        self._last_command_w[env_ids, 0] = 1.0
        self._last_command_w[env_ids, 1] = 0.0
        self._last_command_w[env_ids, 2] = 0.0

    def update_commands(
        self, robot_pos_w: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        k = self.cfg.path_num_points
        all_idx = torch.arange(k, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        # Prune consumed prefix: nearest search only on points ahead of current progress index.
        valid = all_idx >= self.path_idx.unsqueeze(1)
        diffs = self.path_points_w - robot_pos_w.unsqueeze(1)
        d2 = torch.sum(diffs * diffs, dim=-1)
        large = torch.full_like(d2, 1e9)
        nearest_idx = torch.argmin(torch.where(valid, d2, large), dim=1)
        # Monotonic progress along the path (no backward jumps).
        self.path_idx = torch.maximum(self.path_idx, nearest_idx)

        curr_idx = torch.clamp(self.path_idx, max=k - 1)
        next_idx = torch.clamp(curr_idx + 1, max=k - 1)
        # Continuous geometric projection on local segment [curr_idx, curr_idx+1].
        p0 = self.path_points_w[torch.arange(self.num_envs, device=self.device), curr_idx]
        p1 = self.path_points_w[torch.arange(self.num_envs, device=self.device), next_idx]
        seg = p1 - p0
        seg_len_sq = torch.sum(seg * seg, dim=-1, keepdim=True)
        rel = robot_pos_w - p0
        rel_dot_seg = torch.sum(rel * seg, dim=-1, keepdim=True)
        t_raw = rel_dot_seg / seg_len_sq.clamp(min=1e-9)
        t = torch.clamp(t_raw, min=0.0, max=1.0)
        t = torch.where(seg_len_sq > 1e-9, t, torch.zeros_like(t))
        proj = p0 + t * seg
        nearest_dpath = torch.linalg.norm(robot_pos_w - proj, dim=-1, keepdim=True)
        seg_len = torch.sqrt(seg_len_sq.clamp(min=1e-9))
        tangent_xy = seg / seg_len
        # Signed cross-track error: sign from z-component of 2D cross(tangent, robot-projection).
        rel_to_path = robot_pos_w - proj
        cross_z = tangent_xy[:, 0:1] * rel_to_path[:, 1:2] - tangent_xy[:, 1:2] * rel_to_path[:, 0:1]
        nearest_dpath_signed = nearest_dpath * torch.sign(cross_z)
        # Command direction follows path tangent (used for visualization/auxiliary hints).
        cmd_xy = tangent_xy
        commands = torch.zeros((self.num_envs, 3), device=self.device)
        commands[:, :2] = cmd_xy
        cmd_norm = torch.linalg.norm(commands, dim=-1, keepdim=True)
        small = cmd_norm.squeeze(-1) < 1e-3
        commands = torch.where(
            small.unsqueeze(-1),
            self._last_command_w,
            commands / cmd_norm.clamp(min=1e-6),
        )
        self._last_command_w.copy_(commands)
        yaws = torch.atan2(commands[:, 1], commands[:, 0]).unsqueeze(-1)
        # Arc-length coordinate at projection point: s(curr_idx) + t * |segment|.
        s0 = torch.gather(self.path_s, dim=1, index=curr_idx.unsqueeze(1))
        curr_s = s0 + (t * seg_len)
        # Tangent heading of path near current index (used for command direction).
        prev_idx = torch.clamp(curr_idx - 1, min=0)
        next_pts = self.path_points_w[torch.arange(self.num_envs, device=self.device), next_idx]
        prev_pts = self.path_points_w[torch.arange(self.num_envs, device=self.device), prev_idx]
        tangent = next_pts - prev_pts
        tangent_heading = torch.atan2(tangent[:, 1], tangent[:, 0]).unsqueeze(-1)
        # Heading target for reward: bearing to lookahead point taken as last point of local segment.
        lookahead_idx = torch.clamp(curr_idx + (self.cfg.path_segment_len - 1), max=k - 1)
        lookahead_pts = self.path_points_w[torch.arange(self.num_envs, device=self.device), lookahead_idx]
        lookahead_vec = lookahead_pts - robot_pos_w
        lookahead_heading = torch.atan2(lookahead_vec[:, 1], lookahead_vec[:, 0]).unsqueeze(-1)
        small_lookahead = torch.linalg.norm(lookahead_vec, dim=-1, keepdim=True) < 1e-3
        lookahead_heading = torch.where(small_lookahead, tangent_heading, lookahead_heading)
        return (
            commands,
            yaws,
            curr_idx,
            curr_s,
            nearest_dpath,
            nearest_dpath_signed,
            tangent_heading,
            lookahead_heading,
        )

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
