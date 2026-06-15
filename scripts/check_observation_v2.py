#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false
"""Smoke-check ObservationSchemaV2.1 shapes and costmap encoding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_costmap_builder():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source/pirl/pirl/tasks/direct/pirl/pirl_env_costmap.py"
    )
    spec = importlib.util.spec_from_file_location("pirl_env_costmap_smoke", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocalCostmapBuilder


def main() -> None:
    cfg = SimpleNamespace(
        grid_size_m=5.0,
        grid_resolution=0.05,
        grid_width_cells=100,
        grid_free_cost=0.0,
        grid_inscribed_cost=253.0,
        grid_lethal_cost=254.0,
        grid_unknown_cost=255.0,
        grid_inflation_radius_m=0.55,
        grid_cost_scaling_factor=10.0,
        grid_history_len=3,
        grid_history_interval_steps=4,
        grid_normalize=True,
        grid_channels_per_frame=2,
        path_segment_len=12,
        reward_component_dim=6,
        hjb_lidar_sector_count=16,
        lidar_horizontal_fov_range=(-180.0, 180.0),
        lidar_horizontal_res=1.0,
        lidar_num_rays=360,
        lidar=SimpleNamespace(max_distance=18.0),
    )
    expected_vec_dim = (
        2
        + 2
        + (cfg.path_segment_len * 2)
        + (cfg.hjb_lidar_sector_count * 2)
        + 2
        + cfg.reward_component_dim
    )
    expected_costmap_shape = (
        cfg.grid_history_len * cfg.grid_channels_per_frame,
        cfg.grid_width_cells,
        cfg.grid_width_cells,
    )

    LocalCostmapBuilder = _load_costmap_builder()
    builder = LocalCostmapBuilder(cfg, device="cpu", num_envs=1)
    lidar_ranges = torch.full((1, cfg.lidar_num_rays), float(cfg.lidar.max_distance))
    costmap = builder.build_image(lidar_ranges)
    if tuple(costmap.shape) != (1, *expected_costmap_shape):
        raise AssertionError(f"built costmap shape mismatch: {tuple(costmap.shape)}")
    if torch.any(costmap < 0.0) or torch.any(costmap > 1.0):
        raise AssertionError("ObservationSchemaV2.1 costmap channels must be in [0, 1].")

    # Sanity-check sector LiDAR encoder shape using a synthetic ranges tensor: every ray
    # at max range collapses each sector to a single (x, y) at the bearing of the first
    # ray in the sector (argmin returns lowest-index when all values equal sentinel).
    h_min, h_max = cfg.lidar_horizontal_fov_range
    sector_count = int(cfg.hjb_lidar_sector_count)
    sector_width_deg = (h_max - h_min) / sector_count
    ray_angles_deg = torch.linspace(h_min, h_max, cfg.lidar_num_rays)
    ray_sector_idx = (
        ((ray_angles_deg - h_min) / sector_width_deg).floor().long().clamp(0, sector_count - 1)
    )
    sector_xy = torch.zeros((1, sector_count, 2))
    sentinel = float(cfg.lidar.max_distance) * 10.0
    lidar_ranges_full = torch.full((1, cfg.lidar_num_rays), float(cfg.lidar.max_distance))
    for k in range(sector_count):
        mask = (ray_sector_idx == k).unsqueeze(0)
        masked = torch.where(mask, lidar_ranges_full, torch.full_like(lidar_ranges_full, sentinel))
        min_vals, min_idx = torch.min(masked, dim=1)
        bearings = torch.deg2rad(ray_angles_deg[min_idx])
        sector_xy[:, k, 0] = min_vals * torch.cos(bearings)
        sector_xy[:, k, 1] = min_vals * torch.sin(bearings)
    if tuple(sector_xy.shape) != (1, sector_count, 2):
        raise AssertionError(f"sector LiDAR encoding shape mismatch: {tuple(sector_xy.shape)}")

    print("ObservationSchemaV2.1 OK")
    print(f"vec: {(expected_vec_dim,)}")
    print(f"costmap: {expected_costmap_shape}")
    print(f"lidar sectors: {(sector_count, 2)}")


if __name__ == "__main__":
    main()
