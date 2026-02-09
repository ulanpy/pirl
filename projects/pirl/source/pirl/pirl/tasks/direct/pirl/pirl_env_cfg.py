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
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
import math

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
    grid_size_m = 4.0
    grid_resolution = 0.05
    grid_width_cells = int(round(grid_size_m / grid_resolution))
    grid_unknown_value = -1.0
    grid_free_value = 0.0
    grid_occupied_value = 100.0
    grid_inflation_radius_m = 0.1
    grid_history_len = 4
    observation_space = 3 + (grid_width_cells * grid_width_cells * grid_history_len)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = JETTANK_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot_cfg.init_state.pos = (0.0, 0.0, 0.06) 
    
    # sensors
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link", 
        update_period=0.0, 
        history_length=3, 
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Obstacle_.*"]
    )
    
    lidar = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0225)),
        mesh_prim_paths=["/World/envs/env_.*/Obstacle_.*"],
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

    # scene
    # Narrower spacing: 4.0m
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)

    # obstacles
    obstacle_cfg = sim_utils.CylinderCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )

    # controllable joints
    dof_names = [".*_wheel_joint"]

    # action scale
    action_scale = 15.0
    
    # reward scales
    rew_scale_collision = -1.0
    rew_scale_velocity = 2.5
    # discourage driving backwards when rear is not observed
    rew_scale_reverse = -2.0
    # discourage standing still when command exists
    rew_scale_standstill = -0.5
    standstill_speed_threshold = 0.05
    
    # Custom params
    num_obstacles = 6
    obstacle_radius_range = (1.2, 2.0) # Adjusted for 4m spacing
