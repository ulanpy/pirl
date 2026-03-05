from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaacsim.core.prims import SingleXFormPrim


def _scale_from_asset_path(asset_path: str) -> tuple[float, float, float]:
    # ArchVis assets are in centimeters. Convert to meters.
    if "/NVIDIA/Assets/ArchVis/" in asset_path:
        return (0.01, 0.01, 0.01)
    return (1.0, 1.0, 1.0)


def _has_supported_mesh_descendant(root_prim) -> bool:
    """Return True if prim subtree contains geometry trackable by MultiMeshRayCaster."""
    supported = {"Plane", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Mesh"}
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if prim.GetTypeName() in supported:
            return True
        for child in prim.GetChildren():
            stack.append(child)
    return False


def _asset_url_discovery_doc() -> str:
    """How to find new valid obstacle assets.

    1) List a remote prefix:
       curl -s "https://omniverse-content-production.s3-us-west-2.amazonaws.com/?prefix=Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/&max-keys=1000"
    2) Pick `<Key>...*.usd</Key>` entries from the XML.
    3) Prepend:
       https://omniverse-content-production.s3-us-west-2.amazonaws.com/
    """
    return ""


class DynamicObstaclesManager:
    """Runtime moving obstacles with deterministic per-env motion."""

    def __init__(self, cfg, device: torch.device | str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.slot_count = max(0, int(getattr(cfg, "dyn_obstacle_slot_count", 0)))
        self._obstacle_prims: list[list[SingleXFormPrim]] = []
        self._active = torch.zeros((num_envs, self.slot_count), dtype=torch.bool, device=device)
        self._anchor_xy = torch.zeros((num_envs, self.slot_count, 2), dtype=torch.float32, device=device)
        self._radius = torch.zeros((num_envs, self.slot_count), dtype=torch.float32, device=device)
        self._phase = torch.zeros((num_envs, self.slot_count), dtype=torch.float32, device=device)
        self._omega = torch.zeros((num_envs, self.slot_count), dtype=torch.float32, device=device)

    def setup(self) -> None:
        self._obstacle_prims = []
        if not bool(getattr(self.cfg, "dyn_obstacle_enabled", True)):
            return
        if self.slot_count <= 0:
            return
        asset_paths = tuple(getattr(self.cfg, "dyn_obstacle_usd_paths", ()))
        if len(asset_paths) == 0:
            return

        identity_quat = (1.0, 0.0, 0.0, 0.0)
        hidden_pos = (0.0, 0.0, -15.0)

        for env_id in range(self.num_envs):
            ns_path = f"/World/envs/env_{env_id}/GeneratedScene/DynamicObstacles"
            try:
                sim_utils.create_prim(ns_path, prim_type="Xform")
            except ValueError:
                pass

        for slot_idx in range(self.slot_count):
            asset_path = asset_paths[slot_idx % len(asset_paths)]
            scale = _scale_from_asset_path(asset_path)
            slot_prims: list[SingleXFormPrim] = []
            created_paths: list[str] = []
            slot_ok = True
            for env_id in range(self.num_envs):
                prim_path = f"/World/envs/env_{env_id}/GeneratedScene/DynamicObstacles/DynObstacle_{slot_idx}"
                try:
                    sim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type="Xform",
                        translation=hidden_pos,
                        orientation=identity_quat,
                        usd_path=asset_path,
                    )
                    created_paths.append(prim_path)
                    # Lidar/XformPrimView require canonical xform ops [translate, orient, scale].
                    # Use same stage as create_prim (Isaac Lab may use _context.stage, not omni context).
                    stage = sim_utils.get_current_stage()
                    usd_prim = stage.GetPrimAtPath(prim_path)
                    if not usd_prim.IsValid():
                        raise ValueError(f"Spawned prim is invalid: {prim_path}")
                    if not sim_utils.standardize_xform_ops(
                        usd_prim,
                        translation=hidden_pos,
                        orientation=identity_quat,
                        scale=scale,
                    ):
                        raise ValueError(f"Failed to standardize xform ops: {prim_path}")
                    # Invalid/missing USD payload leaves only Xform without mesh; skip such assets.
                    if not _has_supported_mesh_descendant(usd_prim):
                        raise ValueError(f"Asset has no supported mesh descendants: {asset_path}")
                    prim = SingleXFormPrim(prim_path, reset_xform_properties=False)
                    prim.set_local_scale(scale)
                    prim.set_visibility(False)
                    slot_prims.append(prim)
                except Exception:
                    slot_ok = False
                    break
            if not slot_ok:
                for path in created_paths:
                    try:
                        sim_utils.delete_prim(path)
                    except Exception:
                        pass
                continue
            self._obstacle_prims.append(slot_prims)

    def reset(self, env_ids: Sequence[int] | torch.Tensor, env_origins: torch.Tensor) -> None:
        if len(self._obstacle_prims) == 0:
            return
        if len(env_ids) == 0:
            return
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)

        x_range, y_range = getattr(self.cfg, "dyn_obstacle_xy_range", ((-6.0, 6.0), (-6.0, 6.0)))
        keepout = float(getattr(self.cfg, "dyn_obstacle_keepout_radius", 1.5))
        min_sep = float(getattr(self.cfg, "dyn_obstacle_min_separation", 1.2))
        max_tries = int(getattr(self.cfg, "dyn_obstacle_max_sample_tries", 40))
        min_count, max_count = getattr(self.cfg, "dyn_obstacle_count_range", (2, 4))
        max_count = min(int(max_count), len(self._obstacle_prims))
        min_count = min(int(min_count), max_count)
        r_min, r_max = getattr(self.cfg, "dyn_obstacle_motion_radius_range", (0.4, 1.0))
        w_min, w_max = getattr(self.cfg, "dyn_obstacle_motion_speed_range", (0.2, 0.7))
        z_world = float(getattr(self.cfg, "dyn_obstacle_z_world", 0.0))
        margin = float(r_max)
        sx0, sx1 = float(x_range[0]) + margin, float(x_range[1]) - margin
        sy0, sy1 = float(y_range[0]) + margin, float(y_range[1]) - margin
        if sx1 <= sx0:
            sx0, sx1 = x_range
        if sy1 <= sy0:
            sy0, sy1 = y_range

        identity_quat = (1.0, 0.0, 0.0, 0.0)
        for env_id in env_ids_list:
            env_id = int(env_id)
            origin = env_origins[env_id]
            hidden_pos = (float(origin[0]), float(origin[1]), -15.0)
            self._active[env_id, :] = False
            for slot_idx in range(len(self._obstacle_prims)):
                prim = self._obstacle_prims[slot_idx][env_id]
                prim.set_world_pose(position=hidden_pos, orientation=identity_quat)
                prim.set_visibility(False)

            active_count = int(torch.randint(min_count, max_count + 1, (1,), device=self.device).item())
            perm = torch.randperm(len(self._obstacle_prims), device=self.device).tolist()
            placed_xy: list[tuple[float, float]] = []
            activated = 0
            for slot_idx in perm:
                if activated >= active_count:
                    break
                sample_ok = False
                cand_x, cand_y = 0.0, 0.0
                for _ in range(max_tries):
                    cand_x = float(torch.empty(1, device=self.device).uniform_(sx0, sx1).item())
                    cand_y = float(torch.empty(1, device=self.device).uniform_(sy0, sy1).item())
                    if (cand_x * cand_x + cand_y * cand_y) < (keepout * keepout):
                        continue
                    too_close = any((cand_x - px) ** 2 + (cand_y - py) ** 2 < (min_sep * min_sep) for px, py in placed_xy)
                    if not too_close:
                        sample_ok = True
                        break
                if not sample_ok:
                    continue

                r = float(torch.empty(1, device=self.device).uniform_(r_min, r_max).item())
                phase = float(torch.empty(1, device=self.device).uniform_(0.0, 2.0 * math.pi).item())
                omega = float(torch.empty(1, device=self.device).uniform_(w_min, w_max).item())
                if torch.rand(1, device=self.device).item() < 0.5:
                    omega = -omega

                self._anchor_xy[env_id, slot_idx, 0] = cand_x
                self._anchor_xy[env_id, slot_idx, 1] = cand_y
                self._radius[env_id, slot_idx] = r
                self._phase[env_id, slot_idx] = phase
                self._omega[env_id, slot_idx] = omega
                self._active[env_id, slot_idx] = True

                px = float(origin[0]) + cand_x + r * math.cos(phase)
                py = float(origin[1]) + cand_y + r * math.sin(phase)
                yaw = phase + math.pi * 0.5
                half = 0.5 * yaw
                quat_wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
                prim = self._obstacle_prims[slot_idx][env_id]
                prim.set_world_pose(position=(px, py, z_world), orientation=quat_wxyz)
                prim.set_visibility(True)
                placed_xy.append((cand_x, cand_y))
                activated += 1

    def step(self, dt: float, env_origins: torch.Tensor) -> None:
        if len(self._obstacle_prims) == 0:
            return
        if dt <= 0.0:
            return
        z_world = float(getattr(self.cfg, "dyn_obstacle_z_world", 0.0))
        self._phase = self._phase + self._omega * float(dt)

        for env_id in range(self.num_envs):
            ox = float(env_origins[env_id, 0])
            oy = float(env_origins[env_id, 1])
            for slot_idx in range(min(len(self._obstacle_prims), self.slot_count)):
                if not bool(self._active[env_id, slot_idx]):
                    continue
                phase = float(self._phase[env_id, slot_idx])
                r = float(self._radius[env_id, slot_idx])
                ax = float(self._anchor_xy[env_id, slot_idx, 0])
                ay = float(self._anchor_xy[env_id, slot_idx, 1])
                px = ox + ax + r * math.cos(phase)
                py = oy + ay + r * math.sin(phase)
                yaw = phase + math.pi * 0.5
                half = 0.5 * yaw
                quat_wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
                prim = self._obstacle_prims[slot_idx][env_id]
                prim.set_world_pose(position=(px, py, z_world), orientation=quat_wxyz)
