"""Burger embodiment and differential-drive action configuration."""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pirl.robots.burger import BURGER_CFG


@configclass
class BurgerRobotCfg:
    """Asset and fixed action semantics of the TurtleBot3 Burger."""

    articulation: ArticulationCfg = BURGER_CFG.replace(  # type: ignore[attr-defined]
        prim_path="/World/envs/env_.*/Robot",
    )
    articulation.init_state.pos = (0.0, 0.0, 0.02)

    dof_names: list[str] = ["wheel_left_joint", "wheel_right_joint"]
    max_lin_vel: float = 0.22
    max_ang_vel: float = 1.5
    wheel_radius: float = 0.033
    track_width: float = 0.16
