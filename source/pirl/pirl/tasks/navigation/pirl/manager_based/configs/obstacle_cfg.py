"""Dynamic-obstacle pool and reset-time scenario configuration."""

from isaaclab.utils import configclass


@configclass
class DynamicObstacleCfg:
    """Preallocated obstacle capacity and distributions sampled at reset."""

    slot_count: int = 3
    radius: float = 0.25
    height: float = 1.0
    count_range: tuple[int, int] = (1, 3)
    xy_range: tuple[tuple[float, float], tuple[float, float]] = ((-6.0, 6.0), (-6.0, 6.0))
    keepout_radius: float = 0.5
    min_separation: float = 1.5
    motion_radius_range: tuple[float, float] = (0.4, 1.0)
    motion_speed_range: tuple[float, float] = (0.05, 0.2)
    z_world: float = 0.5
