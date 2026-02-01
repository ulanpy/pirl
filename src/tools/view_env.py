import argparse
import os
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="View Jettank Environment with Obstacles.")
# add app launcher arguments
AppLauncher.add_app_launcher_args(parser)
# parse names
args_cli = parser.parse_args()
# launch an omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.extensions import enable_extension

# Enable WebRTC streaming for remote viewing
enable_extension("omni.kit.livestream.webrtc")

from robots.jettank_cfg import JETTANK_CFG, JETTANK_SENSOR_CFG

@configclass
class JettankSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    robot = JETTANK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lidar sensor
    lidar = JETTANK_SENSOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_link",
        mesh_prim_paths=["/World/defaultGroundPlane", "{ENV_REGEX_NS}/Obstacle*"]
    )

    # obstacles - random cuboids
    obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
    )

def main():
    """Main function."""
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # load the scene configuration
    scene_cfg = JettankSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    # Simple simulation loop
    sim_dt = 0.01
    sim_time = 0.0
    count = 0

    print("[INFO]: Simulation setup complete. Use WebRTC client to view.")
    scene.reset()

    while simulation_app.is_running():
        # step the simulation
        scene.write_data_to_sim()
        sim.step()
        scene.read_data_from_sim()
        
        sim_time += sim_dt
        count += 1
        
        if count % 100 == 0:
            if "lidar" in scene.keys():
                lidar_data = scene["lidar"].data.distances
                if lidar_data.numel() > 0:
                    min_dist = torch.min(lidar_data[0])
                    print(f"Scene time: {sim_time:.2f}s | Min Lidar distance: {min_dist:.2f}m")

if __name__ == "__main__":
    main()
    simulation_app.close()
