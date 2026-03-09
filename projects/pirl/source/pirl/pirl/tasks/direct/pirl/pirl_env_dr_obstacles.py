# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization: static obstacle slots (shelves, pallets, cones, etc.) randomized each episode."""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaacsim.core.prims import SingleXFormPrim


def scale_from_asset_path(asset_path: str) -> tuple[float, float, float]:
    """Return (sx, sy, sz) for spawning. ArchVis assets are in cm; scale to meters."""
    if "/NVIDIA/Assets/ArchVis/" in asset_path:
        return (0.01, 0.01, 0.01)
    return (1.0, 1.0, 1.0)


class DomainRandomizationObstacles:
    """Creates a fixed set of obstacle slots and randomizes their poses each episode."""

    def __init__(self, cfg, device: torch.device | str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._obstacle_prims: list[list[SingleXFormPrim]] = []
        self._scene_static_prims: list[list[tuple[SingleXFormPrim, float, float]]] = [
            [] for _ in range(num_envs)
        ]
        self._slot_asset_paths: list[str] = []
        self._slot_half_extents_xy: list[tuple[float, float]] = []
        self._scene_static_names: tuple[str, ...] = tuple(
            getattr(self.cfg, "path_scene_static_obstacle_names", ())
        )
        # World XY of placed obstacles per env (after each reset), for path generation.
        self._last_obstacle_xy: list[list[tuple[float, float]]] = [[] for _ in range(num_envs)]
        # World-frame OBB list per env: (x, y, yaw, hx, hy).
        self._last_obstacle_obb: list[list[tuple[float, float, float, float, float]]] = [
            [] for _ in range(num_envs)
        ]

    def _half_extents_from_asset(self, asset_path: str) -> tuple[float, float]:
        default_hx, default_hy = getattr(
            self.cfg, "dr_obstacle_default_half_extents_xy", (0.45, 0.30)
        )
        safety_scale = float(getattr(self.cfg, "dr_obstacle_half_extents_safety_scale", 1.0))
        overrides = dict(getattr(self.cfg, "dr_obstacle_half_extents_xy_overrides", ()))
        for key, ext in overrides.items():
            if key in asset_path:
                return float(ext[0]) * safety_scale, float(ext[1]) * safety_scale
        return float(default_hx) * safety_scale, float(default_hy) * safety_scale

    def _half_extents_from_scene_name(self, prim_name: str) -> tuple[float, float]:
        default_hx, default_hy = getattr(
            self.cfg, "dr_obstacle_default_half_extents_xy", (0.45, 0.30)
        )
        overrides = dict(
            getattr(self.cfg, "path_scene_static_obstacle_half_extents_xy_overrides", ())
        )
        if prim_name in overrides:
            ext = overrides[prim_name]
            return float(ext[0]), float(ext[1])
        return float(default_hx), float(default_hy)

    @staticmethod
    def _yaw_from_quat_wxyz(quat_wxyz) -> float:
        w = float(quat_wxyz[0])
        x = float(quat_wxyz[1])
        y = float(quat_wxyz[2])
        z = float(quat_wxyz[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def setup(self) -> None:
        """Create obstacle slots (one prim per env per slot). Call after clone_environments."""
        self._obstacle_prims = []
        self._scene_static_prims = [[] for _ in range(self.num_envs)]
        self._slot_asset_paths = []
        self._slot_half_extents_xy = []
        for env_id in range(self.num_envs):
            for prim_name in self._scene_static_names:
                prim_path = f"/World/envs/env_{env_id}/GeneratedScene/{prim_name}"
                try:
                    prim = SingleXFormPrim(prim_path, reset_xform_properties=False)
                    # Validate prim exists in scene by querying pose once.
                    prim.get_world_pose()
                    hx, hy = self._half_extents_from_scene_name(prim_name)
                    self._scene_static_prims[env_id].append((prim, hx, hy))
                except Exception:
                    continue
        asset_paths = tuple(getattr(self.cfg, "dr_obstacle_usd_paths", ()))

        hidden_pos = (0.0, 0.0, -15.0)
        identity_quat = (1.0, 0.0, 0.0, 0.0)

        for env_id in range(self.num_envs):
            ns_path = f"/World/envs/env_{env_id}/GeneratedScene/DomainRandomization"
            try:
                sim_utils.create_prim(ns_path, prim_type="Xform")
            except ValueError:
                pass

        max_slots = int(getattr(self.cfg, "dr_obstacle_slot_count", 0))
        for slot_idx in range(max_slots):
            if len(asset_paths) == 0:
                break
            asset_path = asset_paths[slot_idx % len(asset_paths)]
            scale = scale_from_asset_path(asset_path)
            created_paths: list[str] = []
            slot_prims: list[SingleXFormPrim] = []
            slot_ok = True
            for env_id in range(self.num_envs):
                prim_path = f"/World/envs/env_{env_id}/GeneratedScene/DomainRandomization/Obstacle_{slot_idx}"
                try:
                    sim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type="Xform",
                        translation=hidden_pos,
                        orientation=identity_quat,
                        usd_path=asset_path,
                    )
                    created_paths.append(prim_path)
                    prim = SingleXFormPrim(prim_path, reset_xform_properties=False)
                    prim.set_local_scale(scale)
                    prim.set_visibility(False)
                    slot_prims.append(prim)
                except Exception:
                    slot_ok = False
                    break
            if not slot_ok:
                for p in created_paths:
                    try:
                        sim_utils.delete_prim(p)
                    except Exception:
                        pass
                continue
            self._obstacle_prims.append(slot_prims)
            self._slot_asset_paths.append(asset_path)
            self._slot_half_extents_xy.append(self._half_extents_from_asset(asset_path))

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Hide all slots for env_ids, then place a random subset with valid poses."""
        if len(env_ids) == 0:
            return

        env_ids_list = (
            env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        )
        x_range, y_range = self.cfg.dr_obstacle_xy_range
        keepout = float(self.cfg.dr_obstacle_keepout_radius)
        min_sep = float(self.cfg.dr_obstacle_min_separation)
        max_tries = int(self.cfg.dr_obstacle_max_sample_tries)
        min_count, max_count = self.cfg.dr_obstacle_count_range
        max_count = min(int(max_count), len(self._obstacle_prims))
        min_count = min(int(min_count), max_count)

        for env_id in env_ids_list:
            origin = env_origins[env_id]
            hidden_pos = (float(origin[0]), float(origin[1]), -15.0)
            identity_quat = (1.0, 0.0, 0.0, 0.0)
            for slot_idx in range(len(self._obstacle_prims)):
                prim = self._obstacle_prims[slot_idx][env_id]
                prim.set_world_pose(position=hidden_pos, orientation=identity_quat)
                prim.set_visibility(False)

            active_count = int(
                torch.randint(min_count, max_count + 1, (1,), device=self.device).item()
            )
            active_count = min(active_count, len(self._obstacle_prims))
            perm = torch.randperm(len(self._obstacle_prims), device=self.device).tolist()

            placed_xy: list[tuple[float, float]] = []
            placed_obb: list[tuple[float, float, float, float, float]] = []
            activated = 0
            for slot_idx in perm:
                if activated >= active_count:
                    break
                cand_x, cand_y = 0.0, 0.0
                sample_ok = False
                for _ in range(max_tries):
                    cand_x = float(
                        torch.empty(1, device=self.device).uniform_(x_range[0], x_range[1]).item()
                    )
                    cand_y = float(
                        torch.empty(1, device=self.device).uniform_(y_range[0], y_range[1]).item()
                    )
                    if (cand_x * cand_x + cand_y * cand_y) < (keepout * keepout):
                        continue
                    too_close = any(
                        (cand_x - px) ** 2 + (cand_y - py) ** 2 < (min_sep * min_sep)
                        for px, py in placed_xy
                    )
                    if not too_close:
                        sample_ok = True
                        break
                if not sample_ok:
                    continue

                yaw = float(
                    torch.empty(1, device=self.device).uniform_(-math.pi, math.pi).item()
                )
                half = 0.5 * yaw
                quat_wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
                world_pos = (
                    float(origin[0]) + cand_x,
                    float(origin[1]) + cand_y,
                    0.0,
                )
                prim = self._obstacle_prims[slot_idx][env_id]
                prim.set_world_pose(position=world_pos, orientation=quat_wxyz)
                prim.set_visibility(True)
                placed_xy.append((cand_x, cand_y))
                hx, hy = self._slot_half_extents_xy[slot_idx]
                placed_obb.append(
                    (float(world_pos[0]), float(world_pos[1]), float(yaw), float(hx), float(hy))
                )
                activated += 1

            self._last_obstacle_xy[env_id] = [
                (float(origin[0]) + x, float(origin[1]) + y) for (x, y) in placed_xy
            ]
            if len(self._scene_static_prims[env_id]) == 0:
                for prim_name in self._scene_static_names:
                    prim_path = f"/World/envs/env_{env_id}/GeneratedScene/{prim_name}"
                    try:
                        prim = SingleXFormPrim(prim_path, reset_xform_properties=False)
                        prim.get_world_pose()
                        hx, hy = self._half_extents_from_scene_name(prim_name)
                        self._scene_static_prims[env_id].append((prim, hx, hy))
                    except Exception:
                        continue
            for static_prim, hx, hy in self._scene_static_prims[env_id]:
                try:
                    world_pos, world_quat = static_prim.get_world_pose()
                    static_x = float(world_pos[0])
                    static_y = float(world_pos[1])
                    static_yaw = self._yaw_from_quat_wxyz(world_quat)
                    self._last_obstacle_xy[env_id].append((static_x, static_y))
                    placed_obb.append((static_x, static_y, static_yaw, float(hx), float(hy)))
                except Exception:
                    continue
            self._last_obstacle_obb[env_id] = placed_obb

    def get_obstacle_positions_xy(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> list[torch.Tensor]:
        """Return world XY of placed obstacles for each env (for path generation)."""
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        return [
            torch.tensor(self._last_obstacle_xy[e], device=self.device, dtype=torch.float32)
            if self._last_obstacle_xy[e]
            else torch.zeros(0, 2, device=self.device)
            for e in env_ids
        ]

    def get_obstacle_obbs(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> list[torch.Tensor]:
        """Return obstacle OBBs in world frame for each env as (N, 5): x, y, yaw, hx, hy."""
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        return [
            torch.tensor(self._last_obstacle_obb[e], device=self.device, dtype=torch.float32)
            if self._last_obstacle_obb[e]
            else torch.zeros(0, 5, device=self.device)
            for e in env_ids
        ]
