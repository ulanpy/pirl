import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg

##
# Configuration
##

# Path to the robot's USD file
JETTANK_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "assets", "usd", "jettank.usd"
)

JETTANK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=JETTANK_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=1000.0,
        ),
    },
)

# Lidar sensor configuration
JETTANK_SENSOR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/lidar_link",
    update_period=0.1,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    mesh_prim_paths=["/World/defaultGroundPlane"],
    pattern_cfg=LidarPatternCfg(
        channels=16,
        vertical_fov_range=(-15.0, 15.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=1.0,
    ),
    debug_vis=True,
)
