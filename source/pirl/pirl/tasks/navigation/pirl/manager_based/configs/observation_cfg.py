"""Sensor, costmap, and vector-observation contract configuration."""

import math

import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass


@configclass
class NavigationObservationCfg:
    """Fixed observation function and tensor layout consumed by the policy."""

    lidar_horizontal_fov_range: tuple[float, float] = (-180.0, 180.0)
    lidar_horizontal_res: float = 1.0
    lidar_max_distance: float = 18.0

    grid_size_m: float = 5.0
    grid_resolution: float = 0.05
    grid_free_cost: float = 0.0
    grid_inscribed_cost: float = 253.0
    grid_lethal_cost: float = 254.0
    grid_unknown_cost: float = 255.0
    grid_inflation_radius_m: float = 0.55
    grid_cost_scaling_factor: float = 10.0
    grid_normalize: bool = True
    grid_channels: int = 2

    path_segment_len: int = 12
    reward_component_names: tuple[str, ...] = ("progress", "path_error", "heading", "collision")
    reward_component_obs_clip: float = 1.0

    @property
    def lidar_num_rays(self) -> int:
        h_min, h_max = self.lidar_horizontal_fov_range
        count = math.ceil((h_max - h_min) / self.lidar_horizontal_res) + 1
        return count - 1 if abs(abs(h_max - h_min) - 360.0) < 1e-6 else count

    @property
    def grid_width_cells(self) -> int:
        return int(round(self.grid_size_m / self.grid_resolution))

    @property
    def reward_component_dim(self) -> int:
        return len(self.reward_component_names)

    def lidar_cfg(self) -> MultiMeshRayCasterCfg:
        return MultiMeshRayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/base_scan",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="base",
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/envs/env_.*/GroundPlane", track_mesh_transforms=False
                ),
            ],
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=self.lidar_horizontal_fov_range,
                horizontal_res=self.lidar_horizontal_res,
            ),
            max_distance=self.lidar_max_distance,
            debug_vis=False,
            visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(  # type: ignore[attr-defined]
                prim_path="/Visuals/LidarHits",
                markers={
                    "hit": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    ),
                },
            ),
        )
