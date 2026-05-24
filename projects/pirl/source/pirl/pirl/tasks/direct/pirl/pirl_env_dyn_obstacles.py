"""GPU-batched kinematic obstacles backed by Isaac Lab's RigidObjectCollection.

Design goals:
  * One CylinderCfg per slot, shared across all envs via regex prim paths.
    Geometry is instanced, not replicated per-env, so start-up cost is O(slot_count),
    not O(num_envs * slot_count).
  * Placement sampling, motion integration and pose writes are fully vectorized
    on the device. No Python-per-env loops in hot paths.
  * Integrates with ``isaaclab.scene.InteractiveScene`` lifecycle: the collection
    is registered into ``scene._rigid_object_collections`` so that
    ``scene.write_data_to_sim`` / ``scene.update`` / ``scene.reset`` call it
    automatically.

The public surface mirrors the previous manager for call-site compatibility:
  ``build_collection_cfg(cfg)`` -> ``RigidObjectCollectionCfg``
  ``DynamicObstacles(cfg, device, num_envs).attach(scene)``
  ``obstacles.reset(env_ids, env_origins)`` -- randomize active placements.
  ``obstacles.step(dt, env_origins)``       -- advance circular motion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene

# Sentinel Z used to park inactive cylinders far below the floor without
# affecting the LiDAR (which has a finite max_distance and ray alignment to
# the robot base). Keeping them "somewhere" rather than destroying them lets
# the PhysX rigid-body view keep a fixed, static index layout.
_HIDDEN_Z_WORLD = -50.0


def build_collection_cfg(cfg) -> RigidObjectCollectionCfg:
    """Construct a ``RigidObjectCollectionCfg`` with ``slot_count`` kinematic cylinders.

    One distinct object key ``obstacle_{i}`` maps to one CylinderCfg shared across
    all envs via ``/World/envs/env_.*/DynObstacle_{i}``. Data shape from the
    resulting collection is ``(num_envs, slot_count)``.
    """
    slot_count = int(cfg.dyn_obstacle_slot_count)
    radius = float(cfg.dyn_obstacle_radius)
    height = float(cfg.dyn_obstacle_height)

    cylinder_spawn = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            # Kinematic bodies don't need iteration tuning, but set conservative defaults.
            solver_position_iteration_count=2,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.3, 0.2)),
    )

    rigid_objects: dict[str, RigidObjectCfg] = {}
    for i in range(slot_count):
        rigid_objects[f"obstacle_{i}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/DynObstacle_{i}",
            spawn=cylinder_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, _HIDDEN_Z_WORLD),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


class DynamicObstacles:
    """Kinematic circular-motion obstacles driven by batched pose writes."""

    def __init__(self, cfg, device: torch.device | str, num_envs: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = int(num_envs)
        self.slot_count = int(cfg.dyn_obstacle_slot_count)

        shape = (self.num_envs, self.slot_count)
        self._active = torch.zeros(shape, dtype=torch.bool, device=device)
        self._anchor_xy = torch.zeros((*shape, 2), dtype=torch.float32, device=device)
        self._radius = torch.zeros(shape, dtype=torch.float32, device=device)
        self._phase = torch.zeros(shape, dtype=torch.float32, device=device)
        self._omega = torch.zeros(shape, dtype=torch.float32, device=device)
        # Optional manual control override per (env, slot). When enabled, step()
        # uses the provided local XY/yaw instead of circular-motion integration.
        self._manual_override = torch.zeros(shape, dtype=torch.bool, device=device)
        self._manual_xy = torch.zeros((*shape, 2), dtype=torch.float32, device=device)
        self._manual_yaw = torch.zeros(shape, dtype=torch.float32, device=device)

        self._z_world = float(cfg.dyn_obstacle_z_world)
        self._collection: RigidObjectCollection | None = None

        # Cached helper tensors.
        self._all_env_idx = torch.arange(self.num_envs, device=device, dtype=torch.long)
        self._all_obj_idx = torch.arange(self.slot_count, device=device, dtype=torch.long)

    # ------------------------------------------------------------------ setup

    def attach(self, scene: InteractiveScene, collection_key: str = "dyn_obstacles") -> None:
        """Instantiate the RigidObjectCollection and register it into the scene.

        Must be called after ``scene.clone_environments(...)`` so that the
        ``/World/envs/env_.*`` parents already exist for the regex spawner.
        """
        coll_cfg = build_collection_cfg(self.cfg)
        collection = RigidObjectCollection(coll_cfg)
        scene._rigid_object_collections[collection_key] = collection  # noqa: SLF001
        self._collection = collection

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids: Sequence[int] | torch.Tensor, env_origins: torch.Tensor) -> None:
        """Sample fresh placements for ``env_ids`` and teleport their cylinders.

        Sampling is rejection-free: we draw XY uniformly in the configured arena,
        enforce a keep-out disc around the robot origin and a pairwise
        min-separation via a sequential GPU mask. The result is a boolean
        ``active`` mask, anchor centres, orbit radii and angular speeds per
        (env, slot).
        """
        if self._collection is None:
            raise RuntimeError("DynamicObstacles.attach(scene) must be called before reset.")
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        n = int(env_ids_t.shape[0])
        if n == 0:
            return
        S = self.slot_count

        x_range = self.cfg.dyn_obstacle_xy_range[0]
        y_range = self.cfg.dyn_obstacle_xy_range[1]
        keepout = float(self.cfg.dyn_obstacle_keepout_radius)
        min_sep = float(self.cfg.dyn_obstacle_min_separation)
        min_cnt, max_cnt = self.cfg.dyn_obstacle_count_range
        max_cnt = min(int(max_cnt), S)
        min_cnt = min(int(min_cnt), max_cnt)
        r_min, r_max = self.cfg.dyn_obstacle_motion_radius_range
        w_min, w_max = self.cfg.dyn_obstacle_motion_speed_range
        margin = float(r_max)

        sx0 = float(x_range[0]) + margin
        sx1 = float(x_range[1]) - margin
        sy0 = float(y_range[0]) + margin
        sy1 = float(y_range[1]) - margin
        if sx1 <= sx0:
            sx0, sx1 = float(x_range[0]), float(x_range[1])
        if sy1 <= sy0:
            sy0, sy1 = float(y_range[0]), float(y_range[1])

        # --- 1. Sample candidate XY per (env, slot). ---
        cand = torch.empty((n, S, 2), device=self.device, dtype=torch.float32)
        cand[..., 0].uniform_(sx0, sx1)
        cand[..., 1].uniform_(sy0, sy1)

        # --- 2. Keep-out disc around env origin (local frame). ---
        dist2_origin = (cand * cand).sum(dim=-1)
        origin_ok = dist2_origin >= (keepout * keepout)

        # --- 3. Sequential pairwise min-separation mask on GPU.
        # Go through slots in order; each slot is valid if it's separated from all
        # previously-accepted slots in the same env by >= min_sep.
        valid = torch.zeros((n, S), dtype=torch.bool, device=self.device)
        sep2 = min_sep * min_sep
        for s in range(S):
            if s == 0:
                valid[:, 0] = origin_ok[:, 0]
                continue
            prev_xy = cand[:, :s, :]                    # (n, s, 2)
            cur_xy = cand[:, s, :].unsqueeze(1)         # (n, 1, 2)
            prev_valid = valid[:, :s]                   # (n, s)
            d2 = ((prev_xy - cur_xy) ** 2).sum(dim=-1)  # (n, s)
            # If a previous slot is not valid, ignore its contribution.
            d2 = torch.where(prev_valid, d2, torch.full_like(d2, sep2 + 1.0))
            sep_ok = (d2 >= sep2).all(dim=-1)
            valid[:, s] = origin_ok[:, s] & sep_ok

        # --- 4. Randomize active count per env and keep first K valid slots. ---
        active_counts = torch.randint(
            low=min_cnt, high=max_cnt + 1, size=(n,), device=self.device, dtype=torch.long
        )
        # Permute slot order per env so the "first K valid" selection is random.
        rand_key = torch.rand((n, S), device=self.device)
        perm = rand_key.argsort(dim=-1)                 # (n, S)
        perm_valid = valid.gather(dim=1, index=perm)
        # Prefix-count of valid slots along the permuted order.
        valid_prefix = perm_valid.to(torch.int64).cumsum(dim=-1)
        keep_in_perm = perm_valid & (valid_prefix <= active_counts.unsqueeze(-1))
        # Scatter back to original slot order.
        final_active_local = torch.zeros_like(perm_valid)
        final_active_local.scatter_(dim=1, index=perm, src=keep_in_perm)

        # --- 5. Sample orbit radii, phases, angular speeds. ---
        radii = torch.empty((n, S), device=self.device, dtype=torch.float32).uniform_(float(r_min), float(r_max))
        phases = torch.empty((n, S), device=self.device, dtype=torch.float32).uniform_(0.0, 2.0 * math.pi)
        omegas = torch.empty((n, S), device=self.device, dtype=torch.float32).uniform_(float(w_min), float(w_max))
        sign = torch.where(torch.rand((n, S), device=self.device) < 0.5, -1.0, 1.0)
        omegas = omegas * sign

        # --- 6. Commit per-env state into persistent buffers. ---
        self._active[env_ids_t] = final_active_local
        self._anchor_xy[env_ids_t] = cand
        self._radius[env_ids_t] = radii
        self._phase[env_ids_t] = phases
        self._omega[env_ids_t] = omegas
        # Reset manual overrides for re-initialized envs.
        self._manual_override[env_ids_t] = False
        self._manual_xy[env_ids_t] = 0.0
        self._manual_yaw[env_ids_t] = 0.0

        # --- 7. Compute initial world poses for ALL slots of the affected envs.
        # Active slots placed at anchor+orbit, inactive slots parked at hidden_z.
        local_xy = cand + torch.stack(
            (radii * torch.cos(phases), radii * torch.sin(phases)), dim=-1
        )
        origins = env_origins[env_ids_t, :2]            # (n, 2)
        world_xy = origins.unsqueeze(1) + local_xy       # (n, S, 2)

        z_active = torch.full((n, S), self._z_world, device=self.device)
        z_hidden = torch.full((n, S), _HIDDEN_Z_WORLD, device=self.device)
        z = torch.where(final_active_local, z_active, z_hidden)

        # Yaw = phase + pi/2 (tangent of circle); quaternion wxyz = [cos(yaw/2), 0, 0, sin(yaw/2)].
        yaw = phases + 0.5 * math.pi
        half = 0.5 * yaw
        qw = torch.cos(half)
        qz = torch.sin(half)
        zeros = torch.zeros_like(qw)

        pose = torch.empty((n, S, 7), device=self.device, dtype=torch.float32)
        pose[..., 0] = world_xy[..., 0]
        pose[..., 1] = world_xy[..., 1]
        pose[..., 2] = z
        pose[..., 3] = qw
        pose[..., 4] = zeros
        pose[..., 5] = zeros
        pose[..., 6] = qz

        self._collection.write_object_pose_to_sim(pose, env_ids=env_ids_t)

    # ---------------------------------------------------------- manual control

    def set_manual_obstacle_pose(
        self,
        env_id: int,
        slot_id: int,
        local_xy: torch.Tensor,
        yaw: float,
    ) -> None:
        """Enable manual control of one obstacle slot and set its local pose.

        Args:
            env_id: Environment index [0, num_envs).
            slot_id: Obstacle slot index [0, slot_count).
            local_xy: Local XY coordinates in the env frame (shape: [2]).
            yaw: Obstacle yaw in radians (env frame).
        """
        e = int(env_id)
        s = int(slot_id)
        if e < 0 or e >= self.num_envs:
            raise ValueError(f"env_id out of range: {e}")
        if s < 0 or s >= self.slot_count:
            raise ValueError(f"slot_id out of range: {s}")
        xy = local_xy.to(device=self.device, dtype=torch.float32).reshape(2)
        self._manual_override[e, s] = True
        self._manual_xy[e, s] = xy
        self._manual_yaw[e, s] = float(yaw)

    def clear_manual_obstacle_pose(self, env_id: int, slot_id: int) -> None:
        """Disable manual control override for one obstacle slot."""
        e = int(env_id)
        s = int(slot_id)
        if e < 0 or e >= self.num_envs:
            raise ValueError(f"env_id out of range: {e}")
        if s < 0 or s >= self.slot_count:
            raise ValueError(f"slot_id out of range: {s}")
        self._manual_override[e, s] = False

    # ------------------------------------------------------------------- step

    def step(self, dt: float, env_origins: torch.Tensor) -> None:
        """Advance circular motion and push batched poses into simulation."""
        if self._collection is None or dt <= 0.0:
            return
        # Phase integration (all envs, all slots, vectorized).
        self._phase.add_(self._omega * float(dt))

        local_x = self._anchor_xy[..., 0] + self._radius * torch.cos(self._phase)
        local_y = self._anchor_xy[..., 1] + self._radius * torch.sin(self._phase)
        origins = env_origins[:, :2]                     # (E, 2)
        world_x = origins[:, 0:1] + local_x
        world_y = origins[:, 1:2] + local_y

        z = torch.where(
            self._active,
            torch.full_like(world_x, self._z_world),
            torch.full_like(world_x, _HIDDEN_Z_WORLD),
        )

        yaw = self._phase + 0.5 * math.pi
        half = 0.5 * yaw
        qw = torch.cos(half)
        qz = torch.sin(half)
        zeros = torch.zeros_like(qw)

        if torch.any(self._manual_override):
            manual_world_x = origins[:, 0:1] + self._manual_xy[..., 0]
            manual_world_y = origins[:, 1:2] + self._manual_xy[..., 1]
            manual_half = 0.5 * self._manual_yaw
            manual_qw = torch.cos(manual_half)
            manual_qz = torch.sin(manual_half)
            world_x = torch.where(self._manual_override, manual_world_x, world_x)
            world_y = torch.where(self._manual_override, manual_world_y, world_y)
            z = torch.where(self._manual_override, torch.full_like(z, self._z_world), z)
            qw = torch.where(self._manual_override, manual_qw, qw)
            qz = torch.where(self._manual_override, manual_qz, qz)

        pose = torch.stack((world_x, world_y, z, qw, zeros, zeros, qz), dim=-1)
        self._collection.write_object_pose_to_sim(pose)
