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
import math

@configclass
class PirlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 15.0
    # - spaces definition
    action_space = 2
    observation_space = 3 # dot, cross, forward_speed
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
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.1, 0.0, 0.1)),
        mesh_prim_paths=["/World/envs/env_.*/Obstacle_.*"], 
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=15.0,
        ),
        max_distance=4.0,
        debug_vis=False,
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
    rew_scale_collision = -20.0
    rew_scale_velocity = 1.5
    
    # Custom params
    num_obstacles = 6
    obstacle_radius_range = (1.2, 2.0) # Adjusted for 4m spacing
