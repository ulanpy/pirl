"""Path-generation and robot-reset scenario distributions."""

import math

from isaaclab.utils import configclass


@configclass
class NavigationScenarioCfg:
    """Reset-time navigation scenario distribution; a future curriculum varies these fields."""

    path_length_m: float = 6.0
    path_point_spacing_m: float = 0.10
    path_heading_noise_scale: float = 0.35
    path_mid_turn_rad: float = 0.5
    path_angle_range: tuple[float, float] | None = (-math.pi * 0.5, math.pi * 0.5)
    robot_spawn_radius: float = 0.5
    spawn_angle_range: tuple[float, float] | None = (math.pi * 0.5, math.pi * 1.5)
    path_goal_threshold: float = 0.4

    @property
    def path_num_points(self) -> int:
        return int(round(self.path_length_m / self.path_point_spacing_m)) + 1
