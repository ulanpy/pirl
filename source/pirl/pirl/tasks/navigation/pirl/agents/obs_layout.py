# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flattened offsets for Dict observations (keys in sorted order), aligned with skrl."""

from __future__ import annotations

from typing import Tuple

import gymnasium as gym
from skrl.utils.spaces.torch import compute_space_size


def get_vec_costmap_layout(
    observation_space: gym.Space,
) -> Tuple[int, int, int, Tuple[int, int, int]]:
    """Return slice layout of a flattened Dict(obs) as used by skrl preprocessors.

    skrl flattens ``Dict`` subspaces in **sorted key order**. Sizes use
    ``compute_space_size(..., occupied_size=True)`` so they match the agent's tensors.

    Returns:
        vec_start: start index of ``vec`` in the flat vector
        vec_dim: length of ``vec`` slice
        costmap_start: start index of flattened ``costmap``
        costmap_shape: ``(C, H, W)`` for reshaping the costmap slice
    """
    if not isinstance(observation_space, gym.spaces.Dict):
        raise ValueError("Expected Dict observation space with keys 'vec' and 'costmap'.")
    if "vec" not in observation_space.spaces or "costmap" not in observation_space.spaces:
        raise ValueError("Dict observation space must contain keys 'vec' and 'costmap'.")
    costmap_space = observation_space.spaces["costmap"]
    costmap_shape = costmap_space.shape
    if costmap_shape is None or len(costmap_shape) != 3:
        raise ValueError(f"Expected costmap shape [C,H,W], got {costmap_shape}.")
    c, h, w = int(costmap_shape[0]), int(costmap_shape[1]), int(costmap_shape[2])

    vec_start = -1
    costmap_start = -1
    vec_dim = -1
    offset = 0
    for key in sorted(observation_space.spaces.keys()):
        sub = observation_space.spaces[key]
        size = int(compute_space_size(sub, occupied_size=True))
        if key == "vec":
            vec_start = offset
            vec_dim = size
        elif key == "costmap":
            costmap_start = offset
        offset += size

    if vec_start < 0 or vec_dim < 0 or costmap_start < 0:
        raise ValueError("Failed to infer flattened offsets for vec/costmap.")
    return vec_start, vec_dim, costmap_start, (c, h, w)
