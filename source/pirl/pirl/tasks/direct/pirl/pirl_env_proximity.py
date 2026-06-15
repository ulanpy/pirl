import torch


class ProximityReward:
    """Dense reward penalty based on lidar range in a front sector (proximity to obstacles)."""

    def __init__(self, cfg, device: torch.device | str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._ray_mask = self._build_ray_mask()

    def _build_ray_mask(self) -> torch.Tensor:
        h_min, h_max = self.cfg.lidar_horizontal_fov_range
        num_rays = self.cfg.lidar_num_rays
        if num_rays <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        angles = torch.linspace(h_min, h_max, num_rays, device=self.device)
        half_fov = 0.5 * float(self.cfg.proximity_front_fov_deg)
        return torch.abs(angles) <= half_fov

    def min_selected_range(self, lidar_ranges_m: torch.Tensor) -> torch.Tensor:
        """Return [num_envs, 1] min range in configured proximity FOV."""
        num_rays = lidar_ranges_m.shape[1]
        mask = self._ray_mask[:num_rays]
        selected = (
            lidar_ranges_m
            if (mask.numel() == 0 or not bool(torch.any(mask)))
            else lidar_ranges_m[:, mask]
        )
        return selected.min(dim=1, keepdim=True).values

    def compute_penalty(self, lidar_ranges_m: torch.Tensor | None) -> torch.Tensor:
        """Return [num_envs, 1] penalty. Uses sign from config (should be negative)."""
        if lidar_ranges_m is None:
            return torch.zeros((self.num_envs, 1), device=self.device)

        prox_scale = float(self.cfg.rew_proximity_max_penalty)
        if abs(prox_scale) <= 1e-6:
            return torch.zeros((self.num_envs, 1), device=self.device)

        min_range = self.min_selected_range(lidar_ranges_m)
        activation_distance = float(self.cfg.proximity_activation_distance)
        exp_rate = float(self.cfg.proximity_exponential_rate)
        proximity = torch.clamp(activation_distance - min_range, min=0.0)
        penalty_ratio = 1.0 - torch.exp(-exp_rate * proximity)

        # Keep dense penalty magnitude strictly smaller than terminal collision penalty if it exists.
        collision_scale = abs(float(self.cfg.rew_scale_collision))
        magnitude = abs(prox_scale)
        
        if collision_scale > 1e-3:
            cap = min(magnitude, collision_scale - 1e-6)
        else:
            cap = magnitude
            
        # Return value with original sign
        sign = 1.0 if prox_scale > 0 else -1.0
        return (sign * cap) * penalty_ratio
