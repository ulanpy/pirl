"""Navigation objective and termination thresholds."""

from isaaclab.utils import configclass


@configclass
class NavigationRewardCfg:
    """Base reward profile; curriculum can later supply bounded runtime profiles."""

    progress_scale: float = 10.0
    time_scale: float = -0.002
    success_scale: float = 5.0
    path_error_scale: float = 0.3
    collision_scale: float = -25.0
    heading_scale: float = 0.05
    collision_robot_radius: float = 0.14
