"""Non-physical visual aids for inspecting PIRL episodes in livestream."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def create_heading_markers() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PirlHeadings",
            markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "path_target": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )


def create_path_markers() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PirlPathWindow",
            markers={
                "path": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
    )


def update(
    heading_markers: VisualizationMarkers,
    path_markers: VisualizationMarkers,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    heading_target: torch.Tensor,
    path_points_w: torch.Tensor,
    path_idx: torch.Tensor,
    path_segment_len: int,
) -> None:
    """Draw body heading (cyan), path heading (red), and current path window (green)."""
    num_envs = robot_pos_w.shape[0]
    device = robot_pos_w.device
    marker_pos = robot_pos_w + torch.tensor([0.0, 0.0, 0.5], device=device)
    up_dir = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(num_envs, 1)
    path_orientations = math_utils.quat_from_angle_axis(heading_target.squeeze(-1), up_dir)
    heading_markers.visualize(
        torch.vstack((marker_pos, marker_pos)),
        torch.vstack((robot_quat_w, path_orientations)),
        marker_indices=torch.cat(
            (torch.zeros(num_envs, device=device, dtype=torch.long), torch.ones(num_envs, device=device, dtype=torch.long))
        ),
    )

    segment_idx = path_idx.unsqueeze(1) + torch.arange(path_segment_len, device=device).unsqueeze(0)
    segment_idx = torch.clamp(segment_idx, max=path_points_w.shape[1] - 1)
    segment_xy = path_points_w[torch.arange(num_envs, device=device).unsqueeze(1), segment_idx]
    segment_pos = torch.zeros((num_envs, path_segment_len, 3), device=device)
    segment_pos[..., :2] = segment_xy
    segment_pos[..., 2] = 0.05
    locations = segment_pos.reshape(-1, 3)
    path_markers.visualize(
        locations,
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(locations.shape[0], 1),
        marker_indices=torch.zeros(locations.shape[0], device=device, dtype=torch.long),
    )
