import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def define_markers() -> VisualizationMarkers:
    """Define markers for robot orientation and command direction."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_path_markers() -> VisualizationMarkers:
    """Define markers for local path segment visualization."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pathMarkers",
        markers={
            "path": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def visualize_markers(
    visualization_markers: VisualizationMarkers,
    path_markers: VisualizationMarkers,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    marker_offset: torch.Tensor,
    yaws: torch.Tensor,
    up_dir: torch.Tensor,
    path_points_w: torch.Tensor,
    path_idx: torch.Tensor,
    cfg,
    device: str,
) -> None:
    # Get marker locations and orientations
    marker_locations = robot_pos_w + marker_offset
    forward_orientations = robot_quat_w
    command_orientations = math_utils.quat_from_angle_axis(yaws.squeeze(-1), up_dir)

    # Stack for visualization
    locs = torch.vstack((marker_locations, marker_locations))
    rots = torch.vstack((forward_orientations, command_orientations))

    # Indices: 0 for forward, 1 for command
    indices = torch.hstack(
        (torch.zeros(robot_pos_w.shape[0], device=device, dtype=torch.long),
         torch.ones(robot_pos_w.shape[0], device=device, dtype=torch.long))
    )
    visualization_markers.visualize(locs, rots, marker_indices=indices)

    # Visualize local path segment
    seg_len = cfg.path_segment_len
    seg_indices = path_idx.unsqueeze(1) + torch.arange(seg_len, device=device).unsqueeze(0)
    seg_indices = torch.clamp(seg_indices, max=cfg.path_num_points - 1)
    path_seg_w = path_points_w[torch.arange(robot_pos_w.shape[0], device=device).unsqueeze(1), seg_indices]
    path_seg_w_3 = torch.zeros((robot_pos_w.shape[0], seg_len, 3), device=device)
    path_seg_w_3[:, :, :2] = path_seg_w
    path_seg_w_3[:, :, 2] = 0.05
    locs = path_seg_w_3.reshape(-1, 3)
    rots = torch.tensor([1, 0, 0, 0], device=device).repeat(locs.shape[0], 1)
    indices = torch.zeros(locs.shape[0], device=device, dtype=torch.long)
    path_markers.visualize(locs, rots, marker_indices=indices)
