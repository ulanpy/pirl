import math

import torch
import torch.nn.functional as F


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
            cfg.grid_unknown_value,
            device=device,
        )

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
            self.grid_history[:] = self.cfg.grid_unknown_value
        else:
            self.grid_history[env_ids] = self.cfg.grid_unknown_value

    def build(self, lidar_ranges_m: torch.Tensor) -> torch.Tensor:
        grid = torch.full(
            (self.num_envs, self.grid_width_cells, self.grid_width_cells),
            self.cfg.grid_unknown_value,
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
            grid[env_idx, free_rows[in_bounds], free_cols[in_bounds]] = self.cfg.grid_free_value

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
            grid[env_idx, occ_rows[occ_in_bounds], occ_cols[occ_in_bounds]] = self.cfg.grid_occupied_value

        inflation_cells = int(math.ceil(self.cfg.grid_inflation_radius_m / self.grid_resolution))
        if inflation_cells > 0:
            occ_mask_grid = (grid == self.cfg.grid_occupied_value).float().unsqueeze(1)
            kernel_size = inflation_cells * 2 + 1
            dilated = F.max_pool2d(occ_mask_grid, kernel_size=kernel_size, stride=1, padding=inflation_cells) > 0
            grid = torch.where(dilated.squeeze(1), self.cfg.grid_occupied_value, grid)

        self.grid_history = torch.roll(self.grid_history, shifts=1, dims=1)
        self.grid_history[:, 0] = grid
        return self.grid_history.reshape(self.num_envs, -1)
