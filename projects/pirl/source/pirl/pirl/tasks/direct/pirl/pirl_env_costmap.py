import math

import torch


class LocalCostmapBuilder:
    def __init__(self, cfg, device: str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs

        self.grid_size_m = cfg.grid_size_m
        self.grid_resolution = cfg.grid_resolution
        self.grid_width_cells = cfg.grid_width_cells
        self.grid_half_size = self.grid_size_m / 2.0
        self.grid_history = torch.full(
            (num_envs, cfg.grid_history_len, self.grid_width_cells, self.grid_width_cells),
            cfg.grid_unknown_cost,
            device=device,
        )
        # (num_envs, grid_history_len, 3): world xy + yaw when each frame was captured
        self._pose_history = torch.zeros(
            (num_envs, cfg.grid_history_len, 3),
            device=device,
        )
        self._history_interval = getattr(cfg, "grid_history_interval_steps", 1)
        self._step_count = 0
        # Precompute grid cell centers (meters in base_link frame)
        centers_1d = (torch.arange(self.grid_width_cells, device=device) + 0.5) * self.grid_resolution
        centers_1d = centers_1d - self.grid_half_size
        grid_y, grid_x = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        self.grid_centers = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.rear_mask = (self.grid_centers[:, 0] < 0.0).reshape(self.grid_width_cells, self.grid_width_cells)
        self.rear_count = torch.sum(self.rear_mask).clamp(min=1).item()

        h_min, h_max = cfg.lidar_horizontal_fov_range
        num_angles = math.ceil((h_max - h_min) / cfg.lidar_horizontal_res) + 1
        if abs(abs(h_max - h_min) - 360.0) < 1e-6:
            num_angles -= 1
        lidar_angles = torch.linspace(h_min, h_max, num_angles, device=device)
        if abs(abs(h_max - h_min) - 360.0) < 1e-6:
            lidar_angles = lidar_angles[:-1]
        lidar_angles_rad = torch.deg2rad(lidar_angles)
        self.lidar_cos = torch.cos(lidar_angles_rad)
        self.lidar_sin = torch.sin(lidar_angles_rad)

        self.grid_max_steps = int(math.floor(self.grid_half_size / self.grid_resolution))
        self.grid_step_distances = (
            torch.arange(1, self.grid_max_steps + 1, device=device) * self.grid_resolution
        )

    def reset(self, env_ids):
        if env_ids is None:
            self.grid_history[:] = self.cfg.grid_unknown_cost
            self._pose_history.zero_()
        else:
            self.grid_history[env_ids] = self.cfg.grid_unknown_cost
            self._pose_history[env_ids] = 0.0
        self._step_count = 0

    def build(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build and return flattened costmap history for MLP-style consumers."""
        grid_history_norm = self._build_grid_history(lidar_ranges_m, robot_pos_w, robot_yaw)
        return grid_history_norm.reshape(self.num_envs, -1)

    def build_image(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build and return NCHW ObservationSchemaV2 costmap channels."""
        return self._build_grid_history(lidar_ranges_m, robot_pos_w, robot_yaw)

    def _warp_grid_to_current_frame(
        self,
        grid_past: torch.Tensor,
        pos_past: torch.Tensor,
        yaw_past: torch.Tensor,
        pos_cur: torch.Tensor,
        yaw_cur: torch.Tensor,
    ) -> torch.Tensor:
        """Warp grid_past (one env, H, W) from past body frame to current body frame.
        grid_past: (H, W), pos/yaw past and cur: scalars or 0-dim.
        Returns (H, W) in current frame.
        """
        # Current body cell centers (H*W, 2)
        centers = self.grid_centers
        x_b, y_b = centers[:, 0], centers[:, 1]
        cos_c = torch.cos(yaw_cur).to(x_b.dtype)
        sin_c = torch.sin(yaw_cur).to(x_b.dtype)
        cos_p = torch.cos(yaw_past).to(x_b.dtype)
        sin_p = torch.sin(yaw_past).to(x_b.dtype)
        # Current body -> world
        wx = cos_c * x_b - sin_c * y_b + pos_cur[0]
        wy = sin_c * x_b + cos_c * y_b + pos_cur[1]
        # World -> past body
        dx = wx - pos_past[0]
        dy = wy - pos_past[1]
        x_old = cos_p * dx + sin_p * dy
        y_old = -sin_p * dx + cos_p * dy
        # Past body -> grid indices
        col_f = (x_old + self.grid_half_size) / self.grid_resolution
        row_f = (y_old + self.grid_half_size) / self.grid_resolution
        in_bounds = (
            (row_f >= 0)
            & (row_f < self.grid_width_cells)
            & (col_f >= 0)
            & (col_f < self.grid_width_cells)
        )
        row = row_f.long().clamp(0, self.grid_width_cells - 1)
        col = col_f.long().clamp(0, self.grid_width_cells - 1)
        out_flat = grid_past[row, col].clone()
        out_flat[~in_bounds] = self.cfg.grid_unknown_cost
        return out_flat.reshape(self.grid_width_cells, self.grid_width_cells)

    def _build_grid_history(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = torch.full(
            (self.num_envs, self.grid_width_cells, self.grid_width_cells),
            self.cfg.grid_unknown_cost,
            device=self.device,
        )
        num_rays = min(lidar_ranges_m.shape[1], self.lidar_cos.shape[0])
        lidar_cos = self.lidar_cos[:num_rays]
        lidar_sin = self.lidar_sin[:num_rays]
        step_d = self.grid_step_distances
        x_step = lidar_cos[:, None] * step_d[None, :]
        y_step = lidar_sin[:, None] * step_d[None, :]

        for env_idx in range(self.num_envs):
            ranges = lidar_ranges_m[env_idx, :num_rays]
            # Free space along rays
            free_mask = step_d[None, :] <= ranges[:, None]
            free_x = x_step[free_mask]
            free_y = y_step[free_mask]
            free_cols = torch.floor((free_x + self.grid_half_size) / self.grid_resolution).long()
            free_rows = torch.floor((free_y + self.grid_half_size) / self.grid_resolution).long()
            in_bounds = (
                (free_rows >= 0)
                & (free_rows < self.grid_width_cells)
                & (free_cols >= 0)
                & (free_cols < self.grid_width_cells)
            )
            grid[env_idx, free_rows[in_bounds], free_cols[in_bounds]] = self.cfg.grid_free_cost

            # Occupied cells at ray endpoints (within grid bounds)
            occ_mask = ranges < self.grid_half_size
            occ_x = ranges[occ_mask] * lidar_cos[occ_mask]
            occ_y = ranges[occ_mask] * lidar_sin[occ_mask]
            occ_cols = torch.floor((occ_x + self.grid_half_size) / self.grid_resolution).long()
            occ_rows = torch.floor((occ_y + self.grid_half_size) / self.grid_resolution).long()
            occ_in_bounds = (
                (occ_rows >= 0)
                & (occ_rows < self.grid_width_cells)
                & (occ_cols >= 0)
                & (occ_cols < self.grid_width_cells)
            )
            grid[env_idx, occ_rows[occ_in_bounds], occ_cols[occ_in_bounds]] = self.cfg.grid_lethal_cost

        # Inflation with gradient (Nav2-like)
        if self.cfg.grid_inflation_radius_m > 0.0:
            for env_idx in range(self.num_envs):
                lethal_mask = grid[env_idx] == self.cfg.grid_lethal_cost
                if not torch.any(lethal_mask):
                    continue
                occ_indices = lethal_mask.nonzero(as_tuple=False)
                occ_centers = self.grid_centers[
                    occ_indices[:, 0] * self.grid_width_cells + occ_indices[:, 1]
                ]
                distances = torch.cdist(self.grid_centers, occ_centers).min(dim=1).values
                distances = distances.reshape(self.grid_width_cells, self.grid_width_cells)
                infl_mask = distances <= self.cfg.grid_inflation_radius_m
                # Compute inflation cost (exclude lethal and unknown)
                inflated = self.cfg.grid_inscribed_cost * torch.exp(
                    -self.cfg.grid_cost_scaling_factor * distances
                )
                inflated = torch.clamp(inflated, min=1.0)
                update_mask = (
                    infl_mask
                    & (grid[env_idx] != self.cfg.grid_lethal_cost)
                    & (grid[env_idx] != self.cfg.grid_unknown_cost)
                )
                grid[env_idx] = torch.where(
                    update_mask,
                    torch.maximum(grid[env_idx], inflated),
                    grid[env_idx],
                )

        # Update history buffer only every N steps so that grid_history_len frames span ~1 s
        if self._step_count % self._history_interval == 0:
            self.grid_history = torch.roll(self.grid_history, shifts=1, dims=1)
            self.grid_history[:, 0] = grid
            if robot_pos_w is not None and robot_yaw is not None:
                self._pose_history = torch.roll(self._pose_history, shifts=1, dims=1)
                self._pose_history[:, 0, :2] = robot_pos_w
                y = robot_yaw.squeeze(-1) if robot_yaw.dim() > 1 else robot_yaw
                self._pose_history[:, 0, 2] = y
        self._step_count += 1
        grid_obs = self.grid_history
        if self.cfg.grid_normalize:
            known_mask = (grid_obs != self.cfg.grid_unknown_cost).float()
            cost = torch.where(
                known_mask.bool(),
                grid_obs / self.cfg.grid_lethal_cost,
                torch.tensor(0.0, device=self.device),
            )
            grid_obs = torch.stack((cost, known_mask), dim=2).reshape(
                self.num_envs,
                self.cfg.grid_history_len * self.cfg.grid_channels_per_frame,
                self.grid_width_cells,
                self.grid_width_cells,
            )
        return grid_obs

    def get_danger_score(self) -> torch.Tensor:
        """Return max normalized cost in the current grid (excluding unknown)."""
        grid = self.grid_history[:, 0]
        valid = grid != self.cfg.grid_unknown_cost
        if torch.any(valid):
            norm = grid / self.cfg.grid_lethal_cost
            danger = torch.where(valid, norm, torch.tensor(0.0, device=self.device))
            danger = danger.max(dim=2).values.max(dim=1).values
            return danger.unsqueeze(-1)
        return torch.zeros((self.num_envs, 1), device=self.device)

    def get_unknown_ratio_rear(self) -> torch.Tensor:
        """Return ratio of unknown cells in the rear half of the grid."""
        grid = self.grid_history[:, 0]
        rear_mask = self.rear_mask.to(self.device)
        unknown = (grid == self.cfg.grid_unknown_cost) & rear_mask
        ratio = unknown.sum(dim=(1, 2)) / float(self.rear_count)
        return ratio.unsqueeze(-1)
