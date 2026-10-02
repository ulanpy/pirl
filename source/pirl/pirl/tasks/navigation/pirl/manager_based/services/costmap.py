import math

import torch


class LocalCostmapBuilder:
    """Rasterize LiDAR hit endpoints into a current-frame binary occupancy map."""

    def __init__(self, cfg, device: str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs

        self.grid_size_m = cfg.grid_size_m
        self.grid_resolution = cfg.grid_resolution
        self.grid_width_cells = cfg.grid_width_cells
        self.grid_half_size = self.grid_size_m / 2.0
        h_min, h_max = cfg.lidar_horizontal_fov_range
        num_angles = math.ceil((h_max - h_min) / cfg.lidar_horizontal_res) + 1
        is_full_circle = abs(abs(h_max - h_min) - 360.0) < 1e-6
        lidar_angles = torch.linspace(h_min, h_max, num_angles, device=device)
        if is_full_circle:
            lidar_angles = lidar_angles[:-1]
        lidar_angles_rad = torch.deg2rad(lidar_angles)
        self.lidar_cos = torch.cos(lidar_angles_rad)
        self.lidar_sin = torch.sin(lidar_angles_rad)

    def build(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build and return the flattened current LiDAR hit map."""
        grid_obs = self._build_grid(lidar_ranges_m, robot_pos_w, robot_yaw)
        return grid_obs.reshape(self.num_envs, -1)

    def build_image(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build and return NCHW binary occupancy channel."""
        return self._build_grid(lidar_ranges_m, robot_pos_w, robot_yaw)

    def _build_grid(
        self,
        lidar_ranges_m: torch.Tensor,
        robot_pos_w: torch.Tensor | None = None,
        robot_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        occupied = torch.zeros((self.num_envs, self.grid_width_cells, self.grid_width_cells), device=self.device)
        num_rays = min(lidar_ranges_m.shape[1], self.lidar_cos.shape[0])
        lidar_cos = self.lidar_cos[:num_rays]
        lidar_sin = self.lidar_sin[:num_rays]
        for env_idx in range(self.num_envs):
            ranges = lidar_ranges_m[env_idx, :num_rays]
            # A finite ray endpoint is occupied. Max-range rays do not create
            # an artificial obstacle ring.
            occ_mask = ranges < float(self.cfg.lidar_max_distance)
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
            occupied[env_idx, occ_rows[occ_in_bounds], occ_cols[occ_in_bounds]] = 1.0

        return occupied.unsqueeze(1)
