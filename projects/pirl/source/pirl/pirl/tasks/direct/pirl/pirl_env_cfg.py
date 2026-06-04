# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pirl.robots.burger import BURGER_CFG

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
    # Burger LDS works with full 360-degree visibility.
    lidar_horizontal_fov_range = (-180.0, 180.0)
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
    # Nav2 InflationLayer defaults: inflation_radius=0.55, cost_scaling_factor=10.0
    grid_inflation_radius_m = 0.55  # inflation radius (m); Nav2 default 0.55 (use ~0.15 for tighter inflation)
    grid_cost_scaling_factor = 10.0  # exponential decay; Nav2 default 10.0
    grid_history_len = 3  # number of stacked costmaps (temporal context: CNN sees last K frames as channels)
    # Push a new frame into history every N env steps so that K frames span ~1 s (at 60 env Hz: 4*15=60 steps)
    grid_history_interval_steps = 4
    grid_normalize = True  # normalize costs for RL input
    # ObservationSchemaV2: each history frame becomes [cost, known_mask].
    grid_channels_per_frame = 2
    grid_observation_channels = grid_history_len * grid_channels_per_frame
    # --- Local path observation ---
    # Path is generated once at reset (no replanning inside episode).
    # Controller/reward both use nearest-point anchor on the unconsumed suffix (monotonic prune).
    # Policy receives a fixed-size sliding window of path points in robot frame.
    # Path discretization is metric-controlled: fixed path length + fixed spacing.
    path_length_m = 6.0  # planned path length from start, meters
    path_point_spacing_m = 0.10  # distance between consecutive path points, meters
    path_num_points = int(round(path_length_m / path_point_spacing_m)) + 1  # derived point count
    path_segment_len = 12
    path_goal_threshold = 0.4  # distance to count waypoint as reached
    # Keep reward-history observation shape in sync with enabled reward components.
    reward_component_names = (
        "progress",
        "path_error",
        "heading",
        "proximity",
        "collision",
        "reverse",
    )
    reward_component_dim = len(reward_component_names)
    reward_component_obs_clip = 1.0
    # ObservationSchemaV2.1: K sector-wise LiDAR hit positions in body frame, used by both the
    # policy (extra geometry features for the vec MLP) and HJB (extends Hamiltonian dynamics
    # with body-frame obstacle kinematics, see docs/HJB_THEORY_TIME_DISTANCE.md).
    # Set to 0 to disable and shrink vec back to V2 layout.
    hjb_lidar_sector_count = 16
    hjb_lidar_xy_dim = hjb_lidar_sector_count * 2
    # Curvature (ROS2-like local path: not a straight line).
    path_heading_noise_scale = 0.35  # rad per step; larger → more turns
    path_mid_turn_rad = 0.5  # extra turn in second half of path (rad), ±random
    observation_space = gym.spaces.Dict(
        {
            "vec": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    2 + 2 + (path_segment_len * 2) + hjb_lidar_xy_dim + 2 + reward_component_dim,
                ),
                dtype=np.float32,
            ),
            "costmap": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(grid_observation_channels, grid_width_cells, grid_width_cells),
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
    # Static warehouse scene (shelves + obstacles), no SceneBlox generation.
    sceneblox_usd_paths: tuple[str, ...] = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    )
    # Runtime dynamic obstacles.
    #
    # Implementation: kinematic primitive cylinders driven by a single
    # `RigidObjectCollection` shared across envs. One CylinderCfg is instanced
    # per slot (regex prim_path), poses are written in one batched GPU call per
    # reset/step. LiDAR only reads ranges, so primitive shape is sufficient
    # for obstacle-avoidance learning and avoids the per-prim USD overhead of
    # ArchVis assets (~linear with num_envs × slot_count).
    dyn_obstacle_enabled = True
    dyn_obstacle_slot_count = 20              # distinct cylinders per env (= collection objects)
    dyn_obstacle_count_range = (16, 20)        # active cylinders per episode, clamped to slot_count
    dyn_obstacle_radius = 0.25               # cylinder radius, m
    dyn_obstacle_height = 1.0                # cylinder height, m
    dyn_obstacle_xy_range = ((-6.0, 6.0), (-6.0, 6.0))
    dyn_obstacle_keepout_radius = 0.5        # free disc around env origin (robot spawn zone)
    dyn_obstacle_min_separation = 1.5        # pairwise cylinder separation, m
    dyn_obstacle_motion_radius_range = (0.4, 1.0)
    dyn_obstacle_motion_speed_range = (0.2, 0.8)  # angular speed, rad/s
    dyn_obstacle_z_world = 0.5               # cylinder centre height, m (= height/2 above ground)

    # robot(s)
    robot_cfg: ArticulationCfg = BURGER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot_cfg.init_state.pos = (0.0, 0.0, 0.02)
    
    # sensors
    # Empty scene: MultiMeshRayCaster requires at least one target; use ground so rays can hit floor or max_distance
    lidar = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_scan",
        # Use base_scan pose from URDF directly.
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        # Rays rotate with robot heading/body.
        ray_alignment="base",
        # Таргеты под GeneratedScene. Корень Warehouse_* не трогаем (xform); под ним — только SM_*.
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/GeneratedScene/GroundPlane",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/GeneratedScene/Forklift.*",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/GeneratedScene/SM_.*",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/GeneratedScene/Warehouse_Empty_small_realtime/SM_.*",
                track_mesh_transforms=False,
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

    # Multi-env scene; dynamic obstacles are created under each env namespace.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=35.0, replicate_physics=True)

    # controllable joints (explicit left/right order)
    dof_names = ["wheel_left_joint", "wheel_right_joint"]

    # cmd_vel limits and robot geometry (for wheel speed conversion)
    max_lin_vel = 0.22  # m/s
    max_ang_vel = 2.84  # rad/s
    wheel_radius = 0.033  # m
    track_width = 0.16  # m
    
    # Core reward: r = w1*(s_t - s_{t-1}) - w2*d_path^2 + w3*cos(delta_heading)
    rew_scale_progress = 10.0
    # Cross-track distance penalty coefficient (w2). Applied QUADRATICALLY in _get_rewards()
    # so that small offsets cost almost nothing while large offsets overwhelm the
    # +progress/+heading terms, killing the "drive parallel at fixed offset" exploit.
    # Reference scale: at d=0.5 m penalty is -0.5^2 * 0.3 = -0.075/step (> progress 0.06);
    # at d=0.1 m it is -0.003/step (negligible, doesn't punish normal tracking noise).
    rew_scale_path_error = 0.3
    # Extra safety shaping terms (proximity/collision/reverse) are enabled.
    # Tuned to reduce "freezing" behavior near obstacles while preserving safety pressure.
    rew_scale_collision = -25.0
    collision_robot_radius = 0.14
    proximity_activation_distance = 0.4  # m
    proximity_exponential_rate = 2.0
    proximity_front_fov_deg = 360.0
    rew_proximity_max_penalty = -0.15
    rew_scale_reverse = 0.0
    # Heading alignment coefficient (w3): reward adds w3 * cos(delta_heading) * forward_gate.
    # Gated by forward speed in _get_rewards(), so no bonus for "face path + reverse".
    rew_scale_heading = 0.05
    robot_spawn_radius = 0.5
    spawn_angle_range: tuple[float, float] | None = (math.pi * 0.5, math.pi * 1.5)
    path_angle_range: tuple[float, float] | None = (-math.pi * 0.5, math.pi * 0.5)
