#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false
"""Smoke-check ObservationSchemaV2 shapes and costmap encoding."""

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
        lidar_horizontal_fov_range=(-100.0, 100.0),
        lidar_horizontal_res=1.0,
        lidar_num_rays=201,
        lidar=SimpleNamespace(max_distance=18.0),
    )
    expected_vec_dim = 2 + 2 + (cfg.path_segment_len * 2) + 2 + cfg.reward_component_dim
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
        raise AssertionError("ObservationSchemaV2 costmap channels must be in [0, 1].")

    print("ObservationSchemaV2 OK")
    print(f"vec: {(expected_vec_dim,)}")
    print(f"costmap: {expected_costmap_shape}")


if __name__ == "__main__":
    main()
