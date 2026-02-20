# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pirl.robots.jettank import JETTANK_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
import math
import gymnasium as gym
import numpy as np

@configclass
class PirlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 15.0
    # - spaces definition
    action_space = 2
    # Real EAI G4: 360 deg, 0.28 deg resolution
    # For now, limit rear visibility due to chassis occlusion and reduce rays for faster iteration
    lidar_horizontal_fov_range = (-100.0, 100.0)
    lidar_horizontal_res = 1.0
    lidar_num_rays = math.ceil(
        (lidar_horizontal_fov_range[1] - lidar_horizontal_fov_range[0]) / lidar_horizontal_res
    ) + 1
    if abs(abs(lidar_horizontal_fov_range[1] - lidar_horizontal_fov_range[0]) - 360.0) < 1e-6:
        lidar_num_rays -= 1
    # local costmap (Nav2-like defaults)
    grid_size_m = 5.0  # rolling window size (meters)
    grid_resolution = 0.05  # cell size (meters)
    grid_width_cells = int(round(grid_size_m / grid_resolution))  # grid width/height in cells
    grid_free_cost = 0.0  # Nav2 free space cost
    grid_inscribed_cost = 253.0  # Nav2 inscribed inflated obstacle cost
    grid_lethal_cost = 254.0  # Nav2 lethal obstacle cost
    grid_unknown_cost = 255.0  # Nav2 unknown space cost
    grid_inflation_radius_m = 0.15  # inflation radius (meters)
    grid_cost_scaling_factor = 10.0  # inflation exponential decay factor
    grid_history_len = 4  # number of stacked costmaps (temporal context: CNN sees last K frames as channels)
    # Push a new frame into history every N env steps so that K frames span ~1 s (at 60 env Hz: 4*15=60 steps)
    grid_history_interval_steps = 15
    grid_normalize = True  # normalize costs for RL input
    # local path segment (Nav2-like: controller uses a local slice of the global path)
    # Fewer points spaced ~2 m apart so waypoint bonus is rarer and robot does not abuse it over obstacle avoidance.
    path_num_points = 4  # waypoints along path (spaced ~2 m)
    path_segment_len = 1  # points provided to policy
    path_radius_range = (0.6, 6.0)  # path from 0.6 m to 6 m from env origin → ~2 m between consecutive waypoints
    path_goal_threshold = 0.4  # distance to count waypoint as reached (slightly larger for 2 m spacing)
    observation_space = gym.spaces.Dict(
        {
            "vec": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(5 + (path_segment_len * 2),),
                dtype=np.float32,
            ),
            "costmap": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(grid_history_len, grid_width_cells, grid_width_cells),
                dtype=np.float32,
            ),
        }
    )
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # ground friction
    ground_static_friction = 0.7
    ground_dynamic_friction = 0.7
    ground_friction_combine = "max"

    # robot(s)
    robot_cfg: ArticulationCfg = JETTANK_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot_cfg.init_state.pos = (0.0, 0.0, 0.03) 
    
    # sensors
    lidar = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0225)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle_.*",
                track_mesh_transforms=True,
            ),
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=lidar_horizontal_fov_range,
            horizontal_res=lidar_horizontal_res,
        ),
        # Slightly above real range to ease debugging
        max_distance=18.0,
        debug_vis=False,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(
            prim_path="/Visuals/LidarHits",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        ),
    )

    # scene
    # Keep envs far enough to avoid cross-env obstacle leakage into local costmaps.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=6.0, replicate_physics=True)

    # obstacles
    obstacle_cfg = sim_utils.CylinderCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )

    # controllable joints (explicit left/right order)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # cmd_vel limits and robot geometry (for wheel speed conversion)
    max_lin_vel = 0.5  # m/s
    max_ang_vel = 0.5  # rad/s
    wheel_radius = 0.03  # m (60mm diameter)
    track_width = 0.242  # m
    
    # reward scales
    rew_scale_standstill = 0
    standstill_speed_threshold = 0.05
    spin_rate_threshold = 0.5
    # reward progress toward current path point
    rew_scale_progress = 10.0
    # one-time bonus when reaching each path point
    rew_goal_bonus = 10.0
    # small per-step penalty (must stay much smaller than goal bonus)
    rew_step_penalty = -0.01
    # collision penalty (geometric: robot center vs obstacle centers)
    rew_scale_collision = -10.0
    collision_robot_radius = 0.18  # m, for geometric collision (circle overlap in XY)

    # Custom params
    num_obstacles = 5
    obstacle_radius_range = (1.2, 2.0)  # Adjusted for 4m spacing
    # Robot spawn: random XY in disk of this radius from env origin (inside obstacle ring)
    robot_spawn_radius = 0.5
    # Obstacles move at this speed (m/s); random direction per obstacle per env, bounce at boundary
    obstacle_speed = 0.5
    obstacle_boundary_radius = 2.2  # bounce when distance from env origin exceeds this

    # Typical local avoidance scenario: robot in "start zone", path and obstacles ahead (like real deployment).
    # Angles in radians; 0 = +X, pi/2 = +Y. Set all three to None to restore old "spawn in chaos" behavior.
    # Spawn sector: back half so robot is not dropped in the middle of dynamics.
    spawn_angle_range: tuple[float, float] | None = (math.pi * 0.5, math.pi * 1.5)
    # Path points sector: e.g. (-pi/2, pi/2) = front half so path is ahead of spawn.
    path_angle_range: tuple[float, float] | None = (-math.pi * 0.5, math.pi * 0.5)
    # Obstacles sector: e.g. (-pi/3, pi/3) = front cone so dynamics are ahead, not behind.
    obstacle_angle_range: tuple[float, float] | None = (-math.pi / 3.0, math.pi / 3.0)
