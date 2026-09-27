"""Simulation timing and world-physics configuration."""

from isaaclab.utils import configclass


@configclass
class SimulationControlCfg:
    """Fixed simulator and control-loop semantics for the navigation environment."""

    physics_dt: float = 1 / 120
    decimation: int = 2
    episode_length_s: float = 15.0
    ground_static_friction: float = 0.7
    ground_dynamic_friction: float = 0.7
    ground_friction_combine: str = "max"
