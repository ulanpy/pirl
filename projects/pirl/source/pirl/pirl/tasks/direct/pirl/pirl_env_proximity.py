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

    def compute_penalty(self, lidar_ranges_m: torch.Tensor | None) -> torch.Tensor:
        """Return [num_envs, 1] penalty (<= 0). Uses front-sector min range; capped below collision penalty."""
        if lidar_ranges_m is None:
            return torch.zeros((self.num_envs, 1), device=self.device)

        max_prox_penalty = float(self.cfg.rew_proximity_max_penalty)
        if max_prox_penalty <= 0.0:
            return torch.zeros((self.num_envs, 1), device=self.device)

        ranges = lidar_ranges_m
        num_rays = ranges.shape[1]
        mask = self._ray_mask[:num_rays]
        if mask.numel() == 0 or not bool(torch.any(mask)):
            selected = ranges
        else:
            selected = ranges[:, mask]

        min_range = selected.min(dim=1, keepdim=True).values
        activation_distance = float(self.cfg.proximity_activation_distance)
        exp_rate = float(self.cfg.proximity_exponential_rate)
        proximity = torch.clamp(activation_distance - min_range, min=0.0)
        penalty_ratio = 1.0 - torch.exp(-exp_rate * proximity)

        collision_cap = max(abs(float(self.cfg.rew_scale_collision)) - 1e-6, 0.0)
        penalty_cap = min(max_prox_penalty, collision_cap)
        return -penalty_cap * penalty_ratio
