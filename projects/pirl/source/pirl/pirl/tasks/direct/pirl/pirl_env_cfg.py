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
    # Nav2 InflationLayer defaults: inflation_radius=0.55, cost_scaling_factor=10.0
    grid_inflation_radius_m = 0.55  # inflation radius (m); Nav2 default 0.55 (use ~0.15 for tighter inflation)
    grid_cost_scaling_factor = 10.0  # exponential decay; Nav2 default 10.0
    grid_history_len = 4  # number of stacked costmaps (temporal context: CNN sees last K frames as channels)
    # Push a new frame into history every N env steps so that K frames span ~1 s (at 60 env Hz: 4*15=60 steps)
    grid_history_interval_steps = 4
    grid_normalize = True  # normalize costs for RL input
    # local path segment (Nav2-like: controller uses a local slice of the global path)
    # Fewer points spaced ~2 m apart so waypoint bonus is rarer and robot does not abuse it over obstacle avoidance.
    path_num_points = 4  # waypoints along path (spaced ~2 m)
    path_segment_len = 5  # points provided to policy
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
    # Static warehouse scene (shelves + obstacles), no SceneBlox generation.
    sceneblox_usd_paths: tuple[str, ...] = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    )
    # Per-episode static obstacle randomization inside the warehouse.
    dr_obstacle_usd_paths: tuple[str, ...] = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Forklift/forklift.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Industrial/Racks/RackLarge_A5.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Industrial/Racks/RackLong_A7.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Industrial/Racks/RackLong_A5.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Industrial/Racks/RackLong_A4.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_02.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_02.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    )
    dr_obstacle_slot_count = 24
    dr_obstacle_count_range = (8, 18)
    dr_obstacle_xy_range = ((-10.0, 10.0), (-10.0, 10.0))
    dr_obstacle_keepout_radius = 1.8
    dr_obstacle_min_separation = 2.0
    dr_obstacle_max_sample_tries = 40
    # Runtime dynamic obstacles (stable replacement for people in RL training).
    dyn_obstacle_enabled = True
    # Variety of non-trivial props (chairs/storage/table) for lidar obstacle perception.
    # ArchVis assets are in centimeters; DynamicObstaclesManager scales them to meters.
    # How to discover valid USD URLs later:
    #   curl -s "https://omniverse-content-production.s3-us-west-2.amazonaws.com/?prefix=Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/&max-keys=1000"
    # Then pick <Key>...*.usd</Key> entries and prepend:
    #   https://omniverse-content-production.s3-us-west-2.amazonaws.com/
    dyn_obstacle_usd_paths: tuple[str, ...] = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Seating/Jobba/Jobba_Chair.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Seating/Petite_Chair.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Seating/Stackable.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Seating/Caprice/Caprice_A.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Storage/Contemporary/Contemporary_StorageCube.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Storage/Standard/Standard_SmallUnit.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Tables/OakTableSmall.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Commercial/Tables/Kettle.usd",
    )
    dyn_obstacle_slot_count = 16
    dyn_obstacle_count_range = (6, 12)
    dyn_obstacle_xy_range = ((-8.0, 8.0), (-8.0, 8.0))
    dyn_obstacle_keepout_radius = 1.5
    dyn_obstacle_min_separation = 1.5
    dyn_obstacle_max_sample_tries = 40
    dyn_obstacle_motion_radius_range = (0.4, 1.0)
    dyn_obstacle_motion_speed_range = (0.2, 0.8)  # angular speed, rad/s
    dyn_obstacle_z_world = 0.03

    # robot(s)
    robot_cfg: ArticulationCfg = JETTANK_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot_cfg.init_state.pos = (0.0, 0.0, 0.03) 
    
    # sensors
    # Empty scene: MultiMeshRayCaster requires at least one target; use ground so rays can hit floor or max_distance
    lidar = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        # Только yaw: иначе при наклоне робота передние лучи уходят вниз и бьют в пол (GroundPlane) на 2–3 м.
        ray_alignment="yaw",
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
        debug_vis=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=35.0, replicate_physics=True)

    # controllable joints (explicit left/right order)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # cmd_vel limits and robot geometry (for wheel speed conversion)
    max_lin_vel = 0.5  # m/s
    max_ang_vel = 3.0  # rad/s
    wheel_radius = 0.03  # m (60mm diameter)
    track_width = 0.242  # m
    
    # reward progress toward current path point
    rew_scale_progress = 10.0
    # one-time bonus when reaching each path point
    rew_goal_bonus = 10.0
    # small per-step penalty (must stay much smaller than goal bonus)
    rew_step_penalty = 0.0
    rew_scale_collision = -50.0
    collision_robot_radius = 0.18
    # Dense proximity penalty (exp-shaped); 0 with no obstacles.
    # Penalty activates when nearest obstacle in front sector is closer than this distance.
    proximity_activation_distance = 0.8  # m
    # Larger value -> steeper growth as obstacle gets closer.
    proximity_exponential_rate = 2.0
    # FOV for proximity penalty. Set to 360 to cover all sides.
    proximity_front_fov_deg = 360.0
    # Absolute cap for proximity penalty magnitude (should be negative).
    rew_proximity_max_penalty = -0.05
    # Additional penalty on positive closing speed (m/s) near obstacles.
    rew_scale_proximity_rate = -0.05
    # Apply proximity-rate penalty only when nearest range is below this distance.
    proximity_rate_gate_distance = 1.2
    # Optional anti-reverse shaping (0 disables). Applies as: scale * relu(-forward_speed).
    rew_scale_reverse = -0.5
    rew_scale_heading = 0.01
    # Penalty for yaw-command jitter (squared difference of normalized yaw action)
    rew_scale_action_rate = -0.25

    robot_spawn_radius = 0.5
    spawn_angle_range: tuple[float, float] | None = (math.pi * 0.5, math.pi * 1.5)
    path_angle_range: tuple[float, float] | None = (-math.pi * 0.5, math.pi * 0.5)
