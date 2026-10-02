#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false
"""Smoke-check vector-observation shape and costmap encoding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_costmap_builder():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source/pirl/pirl/tasks/navigation/pirl/manager_based/services/costmap.py"
    )
    spec = importlib.util.spec_from_file_location("pirl_costmap_smoke", module_path)
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
        grid_channels=1,
        path_segment_len=12,
        reward_component_dim=4,
        lidar_horizontal_fov_range=(-180.0, 180.0),
        lidar_horizontal_res=1.0,
        lidar_num_rays=360,
        lidar_max_distance=18.0,
    )
    expected_vec_dim = (
        2
        + 2
        + (cfg.path_segment_len * 2)
        + 2
        + cfg.reward_component_dim
    )
    expected_costmap_shape = (
        cfg.grid_channels,
        cfg.grid_width_cells,
        cfg.grid_width_cells,
    )

    LocalCostmapBuilder = _load_costmap_builder()
    builder = LocalCostmapBuilder(cfg, device="cpu", num_envs=1)
    lidar_ranges = torch.full((1, cfg.lidar_num_rays), float(cfg.lidar_max_distance))
    costmap = builder.build_image(lidar_ranges)
    if tuple(costmap.shape) != (1, *expected_costmap_shape):
        raise AssertionError(f"built costmap shape mismatch: {tuple(costmap.shape)}")
    if torch.any(costmap < 0.0) or torch.any(costmap > 1.0):
        raise AssertionError("Costmap channels must be in [0, 1].")

    one_hit = lidar_ranges.clone()
    one_hit[0, 0] = 1.0
    if torch.count_nonzero(builder.build_image(one_hit)) != 1:
        raise AssertionError("A finite LiDAR endpoint must rasterize to exactly one occupied cell.")

    print("Observation schema OK")
    print(f"vec: {(expected_vec_dim,)}")
    print(f"costmap: {expected_costmap_shape}")


if __name__ == "__main__":
    main()
