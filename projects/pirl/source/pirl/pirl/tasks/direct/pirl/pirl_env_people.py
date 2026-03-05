from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaacsim.core.prims import SingleXFormPrim


class PeopleManager:
    """Runtime people spawner/controller for dynamic obstacle behavior.

    Uses IRA (omni.anim.people + isaacsim.replicator.agent) to spawn characters and
    inject GoTo commands. Created because full IRA workflow (actor_sdg.py + YAML) is
    designed for data generation, not RL: IRA runs fixed-duration sims and has no
    per-episode reset. PeopleManager integrates with Isaac Lab's reset/step loop.
    See docs/PEOPLE_IRA_WORKFLOW.md for full IRA alternative.

    Implementation notes (why characters stay on floor):

    1. _enforce_z_override — Z override in session layer
       AnimGraph/BehaviorScript writes character position to session layer (anon:World0-session.usda).
       Our set_world_pose writes to root layer, so animation overwrote it.
       Fix: each step() we switch edit target to session layer via
       omni.usd.set_edit_target_by_identifier(stage, session.identifier), find the driven prim
       (ManRoot/asset_name with xformOp:translate), take current XY from navigation, replace Z with
       target_z=0.95, and write translate to session layer so we override AnimGraph.

    2. _find_driven_xform_path — find the driven prim
       Animation drives a nested prim (ManRoot/female_adult_business_02), not the root.
       We traverse the subtree and find the prim with "ManRoot" in path and xformOp:translate.

    3. people_floor_z_world = 0.0
       Warehouse floor is at Z=0 (same as dr_obstacles). target_z = spawn_z = floor_z.

    4. people_character_parent_path
       All characters under /World/Characters (IRA compatible). For multi-env, we spawn
       num_envs*slots_per_env people and place them at env_origins so each env has its group.

    5. IRA setup on first reset
       setup_after_sim_start runs on first reset(), not in __init__, so NavMesh bakes after
       the scene is fully loaded.

    Key: AnimGraph writes to session layer, we wrote to root. Switching edit target to
    session layer and writing Z each frame lets us override animation and keep characters
    at the correct height.
    """

    def __init__(self, cfg, device: torch.device | str) -> None:
        self.cfg = cfg
        self.device = device
        self._num_envs = 1
        self._slots_per_env = 0
        self._root_prims: list[SingleXFormPrim] = []
        self._root_paths: list[str] = []
        self._control_prims: list[SingleXFormPrim] = []
        self._control_paths: list[str] = []
        self._agent_names: list[str] = []
        self._base_quat_wxyz: list[tuple[float, float, float, float]] = []
        self._env_id_for_slot: list[int] = []  # flat_idx -> env_id
        self._active_slots: set[int] = set()
        self._agent_manager = None
        self._stage_util = None
        self._commands_enabled = False
        self._command_refresh_accum_s = 0.0
        self._navmesh_ready = False
        self._ira_setup_pending = False
        self._ira_setup_done = False

    def _enable_extensions(self) -> None:
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("omni.kit.scripting", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.bundle", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.schema", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.bundle", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.schema", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.people", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)

    @staticmethod
    def _spin_app_updates(num_frames: int = 8) -> None:
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(max(1, int(num_frames))):
            app.update()

    @staticmethod
    def _force_register_anim_graph_schema() -> None:
        """Force USD schema plugin registration for AnimGraph in problematic 5.1 installs."""
        import os

        import omni.kit.app
        from pxr import Plug

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        schema_ext_path = ext_mgr.get_extension_path_by_module("omni.anim.graph.schema")
        if not schema_ext_path:
            return
        schema_resource_path = os.path.join(
            schema_ext_path, "plugins", "AnimGraphSchema", "resources"
        )
        Plug.Registry().RegisterPlugins(schema_resource_path)

    @staticmethod
    def _configure_people_settings(cfg=None, num_envs: int = 1) -> None:
        import carb
        from omni.anim.people.settings import PeopleSettings

        people_root = str(getattr(cfg, "people_character_parent_path", "/World/Characters")) if cfg else "/World/Characters"
        settings = carb.settings.get_settings()
        settings.set(PeopleSettings.NAVMESH_ENABLED, True)
        settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, True)
        settings.set(PeopleSettings.CHARACTER_PRIM_PATH, people_root)
        settings.set("/exts/isaacsim.replicator.agent/skip_biped_setup", False)
        settings.set("/exts/isaacsim.replicator.agent/characters_parent_prim_path", people_root)

    @staticmethod
    def _ensure_navmesh_volume(num_envs: int = 1) -> None:
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        nav_volume_paths: list[str] = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "NavMeshVolume":
                nav_volume_paths.append(str(prim.GetPrimPath()))

        # Create one NavMeshVolume per env if we have fewer than needed.
        for env_id in range(num_envs):
            if env_id < len(nav_volume_paths):
                continue
            parent = f"/World/envs/env_{env_id}" if stage.GetPrimAtPath(f"/World/envs/env_{env_id}").IsValid() else "/World"
            if parent == "/World":
                break
            omni.kit.commands.execute(
                "CreateNavMeshVolumeCommand",
                parent_prim_path=Sdf.Path(parent),
                position=Gf.Vec3d(0.0, 0.0, 0.0),
            )
        nav_volume_paths = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "NavMeshVolume":
                nav_volume_paths.append(str(prim.GetPrimPath()))

        # Scale each volume to cover warehouse floor area. Z=80 ensures floor is included.
        for path in nav_volume_paths[:num_envs]:
            xform = Gf.Matrix4d(1.0)
            xform.SetScale(Gf.Vec3d(80.0, 80.0, 80.0))
            omni.kit.commands.execute(
                "TransformPrim",
                path=Sdf.Path(path),
                new_transform_matrix=xform,
            )

    @staticmethod
    def _wait_for_navmesh_ready(max_frames: int = 240) -> bool:
        import omni.anim.navigation.core as nav

        nav_iface = nav.acquire_interface()
        try:
            nav_iface.start_navmesh_baking_and_wait()
        except Exception:
            return False
        navmesh = nav_iface.get_navmesh()
        return navmesh is not None

    @staticmethod
    def _find_skelroot_prim(stage, root_prim):
        """Find actual SkelRoot under root; CharacterUtil may return non-SkelRoot (e.g. biped_demo_meters)."""
        from pxr import Usd, UsdSkel

        for prim in Usd.PrimRange(root_prim):
            if UsdSkel.Root(prim):
                return prim
        return None

    @staticmethod
    def _get_biped_asset_path(asset_root: str) -> str | None:
        """Return path to Biped_Setup.usd for IRA AnimGraph compatibility."""
        import omni.client

        root = asset_root.rstrip("/")
        result, items = omni.client.list(root)
        if result != omni.client.Result.OK:
            return None
        for item in items:
            name = getattr(item, "relative_path", "") or ""
            if name and (name == "Biped_Setup.usd" or name.lower().endswith("biped_setup.usd")):
                return f"{root}/{name}"
        # Fallback: standard Isaac path
        return f"{root}/Biped_Setup.usd"

    @staticmethod
    def _list_character_usd_assets(asset_root: str, use_biped: bool = False) -> list[str]:
        import omni.client

        if use_biped:
            biped = PeopleManager._get_biped_asset_path(asset_root)
            return [biped] if biped else []

        result, folder_list = omni.client.list(asset_root)
        if result != omni.client.Result.OK:
            return []

        assets: list[str] = []
        for folder in folder_list:
            if not (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN):
                continue
            folder_name = folder.relative_path
            if folder_name.startswith("."):
                continue
            folder_path = f"{asset_root.rstrip('/')}/{folder_name}"
            sub_result, file_list = omni.client.list(folder_path)
            if sub_result != omni.client.Result.OK:
                continue
            for item in file_list:
                rel = item.relative_path
                if rel.lower().endswith((".usd", ".usda")):
                    assets.append(f"{folder_path}/{rel}")
                    break
        return assets

    def _ensure_command_file(self) -> str | None:
        """Create minimal command file for behavior script; return path or None."""
        import tempfile

        try:
            fd, path = tempfile.mkstemp(suffix=".txt", prefix="pirl_people_cmd_")
            with open(fd, "w") as f:
                for name in getattr(self, "_agent_names", []):
                    f.write(f"{name} Idle 1\n")
            return path
        except Exception:
            return None

    def setup(self, num_envs: int = 1, env_origins=None) -> None:
        if not bool(getattr(self.cfg, "people_enabled", False)):
            return

        self._num_envs = max(1, int(num_envs))
        self._setup_env_origins = env_origins
        # All characters under /World/Characters (IRA compatible). For multi-env, we spawn many people
        # and place them at env_origins so each env has its own group walking in that warehouse.
        self._enable_extensions()
        # Force-load schema registration module before any AnimationGraph API command.
        import omni.anim.graph.schema  # noqa: F401
        import AnimGraphSchema  # noqa: F401
        self._force_register_anim_graph_schema()
        self._spin_app_updates(12)
        self._configure_people_settings(self.cfg, self._num_envs)
        self._ensure_navmesh_volume(self._num_envs)
        self._spin_app_updates(12)
        self._navmesh_ready = self._wait_for_navmesh_ready()

        from isaacsim.replicator.agent.core.agent_manager import AgentManager
        from isaacsim.replicator.agent.core.settings import AssetPaths
        from isaacsim.replicator.agent.core.stage_util import CharacterUtil

        self._agent_manager = AgentManager.get_instance()
        self._stage_util = CharacterUtil

        asset_root = getattr(self.cfg, "people_asset_root", None) or AssetPaths.default_character_path()
        if not asset_root:
            return
        use_biped = bool(getattr(self.cfg, "people_use_biped_asset", False))
        assets = self._list_character_usd_assets(asset_root, use_biped=use_biped)
        if len(assets) == 0:
            return
        force_single_asset = bool(getattr(self.cfg, "people_force_single_asset", False))
        if force_single_asset and not use_biped:
            assets = [assets[0]]

        use_anim_graph = bool(getattr(self.cfg, "people_use_anim_graph", False))
        slots_per_env = int(getattr(self.cfg, "people_slot_count", 0))
        if slots_per_env <= 0:
            return
        self._slots_per_env = slots_per_env

        identity_quat = (1.0, 0.0, 0.0, 0.0)
        skelroots = []
        self._root_prims = []
        self._root_paths = []
        self._control_prims = []
        self._control_paths = []
        self._agent_names = []
        self._base_quat_wxyz = []
        self._env_id_for_slot = []

        parent_path = str(getattr(self.cfg, "people_character_parent_path", "/World/Characters"))
        try:
            sim_utils.create_prim(parent_path, prim_type="Xform")
        except ValueError:
            pass

        total_slots = self._num_envs * slots_per_env
        for i in range(total_slots):
            env_id = i // slots_per_env
            slot_idx = i % slots_per_env
            name = f"Character_{i}"
            usd_path = assets[i % len(assets)]
            prim_path = f"{parent_path}/{name}"
            try:
                sim_utils.create_prim(
                    prim_path=prim_path,
                    prim_type="Xform",
                    translation=(0.0, 0.0, 0.0),
                    orientation=identity_quat,
                    usd_path=usd_path,
                )
            except Exception:
                continue

            import omni.usd
            from pxr import UsdSkel

            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(prim_path)
            if not root_prim.IsValid():
                continue
            prim_wrapper = SingleXFormPrim(prim_path, reset_xform_properties=False)
            mesh_count = self._count_subtree_meshes(prim_path)
            min_mesh_count = int(getattr(self.cfg, "people_min_mesh_count", 1))
            if mesh_count < min_mesh_count:
                continue
            skelroot = CharacterUtil.get_character_skelroot_by_root(root_prim)
            if skelroot is None:
                continue
            # Biped: CharacterUtil may return biped_demo_meters (not SkelRoot). Find actual SkelRoot.
            if stage and not UsdSkel.Root(skelroot):
                real_skel = PeopleManager._find_skelroot_prim(stage, root_prim)
                if real_skel is not None:
                    skelroot = real_skel
            skelroots.append(skelroot)
            control_path = str(skelroot.GetPrimPath())
            control_wrapper = SingleXFormPrim(control_path, reset_xform_properties=False)
            if self._setup_env_origins is not None and env_id < len(self._setup_env_origins):
                o = self._setup_env_origins[env_id]
                hidden_pos = (float(o[0]), float(o[1]), -15.0)
            else:
                hidden_pos = (0.0, 0.0, -15.0)
            prim_wrapper.set_world_pose(position=hidden_pos, orientation=identity_quat)
            prim_wrapper.set_visibility(False)
            self._set_subtree_visibility(prim_path, False)
            visual_scale = float(getattr(self.cfg, "people_debug_visual_scale", 1.0))
            self._set_root_uniform_scale(prim_path, visual_scale)
            # Base orientation: Biped_Setup.usd may be Y-up; Isaac Sim is Z-up. Rotate around X to stand.
            if use_biped:
                deg = float(getattr(self.cfg, "people_biped_stand_rotation_deg", -90.0))
                half = 0.5 * math.radians(deg)
                self._base_quat_wxyz.append((math.cos(half), math.sin(half), 0.0, 0.0))
            else:
                self._base_quat_wxyz.append((1.0, 0.0, 0.0, 0.0))
            self._root_prims.append(prim_wrapper)
            self._root_paths.append(prim_path)
            self._control_prims.append(control_wrapper)
            self._control_paths.append(control_path)
            self._agent_names.append(name)
            self._env_id_for_slot.append(env_id)

        # IRA setup (AnimGraph + Agent Manager) deferred until after sim.reset().
        # Store flag so setup_after_sim_start() can run it when sim is playing.
        self._ira_setup_pending = use_anim_graph and len(skelroots) > 0
        self._commands_enabled = False

    def setup_after_sim_start(self, sim=None) -> None:
        """Run IRA setup (AnimGraph + Agent Manager) after sim.reset()."""
        if not self._ira_setup_pending or len(self._root_paths) == 0:
            return
        self._ira_setup_pending = False
        self._ira_setup_done = True
        self._spin_app_updates(12)
        # Re-bake NavMesh after sim.reset() so warehouse geometry is fully composed.
        self._ensure_navmesh_volume(self._num_envs)
        self._spin_app_updates(8)
        self._navmesh_ready = self._wait_for_navmesh_ready()
        try:
            import carb
            import omni.kit.commands as okc
            import NavSchema
            from pxr import Sdf

            from isaacsim.replicator.agent.core.stage_util import CharacterUtil

            def _run() -> None:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    return
                # Use root layer for overrides; session layer may not compose correctly in Isaac Lab.
                root = stage.GetRootLayer()
                if root:
                    stage.SetEditTarget(root)
                for prim_path in self._root_paths:
                    okc.execute(
                        "ApplyNavMeshAPICommand", prim_path=prim_path, api=NavSchema.NavMeshExcludeAPI
                    )
                cmd_file = self._ensure_command_file()
                if cmd_file:
                    carb.settings.get_settings().set(
                        "/exts/omni.anim.people/command_settings/command_file_path", cmd_file
                    )
                for i, root_path in enumerate(self._root_paths):
                    self._set_subtree_visibility(root_path, True)
                    self._set_visibility(root_path, True)
                    if i < len(self._control_paths):
                        self._set_visibility(self._control_paths[i], True)
                self._spin_app_updates(4)
                sim_mgr = __import__(
                    "isaacsim.replicator.agent.core.simulation", fromlist=["SimulationManager"]
                ).SimulationManager()
                sim_mgr.setup_all_characters()
                biped = CharacterUtil.get_default_biped_character()
                anim_graph = CharacterUtil.get_anim_graph_from_character(biped) if biped and biped.IsValid() else None
                if anim_graph and anim_graph.IsValid():
                    anim_path = Sdf.Path(str(anim_graph.GetPrimPath()))
                    for cp in self._control_paths:
                        try:
                            okc.execute("RemoveAnimationGraphAPICommand", paths=[Sdf.Path(cp)])
                        except Exception:
                            pass
                        okc.execute(
                            "ApplyAnimationGraphAPICommand",
                            paths=[Sdf.Path(cp)],
                            animation_graph_path=anim_path,
                        )
                    self._spin_app_updates(8)
                self._hide_biped_template(biped)
                # Keep visible longer so BehaviorScript can init and register to Agent Manager.
                self._spin_app_updates(48)
                for root_path in self._root_paths:
                    self._set_subtree_visibility(root_path, False)
                    self._set_visibility(root_path, False)
                for cp in self._control_paths:
                    self._set_visibility(cp, False)

            # Use default stage (omni.usd context); sim.get_initial_stage() can differ in Isaac Lab.
            _run()
            self._commands_enabled = True
            self._spin_app_updates(24)
        except Exception:
            self._commands_enabled = False

    @staticmethod
    def _set_subtree_visibility(root_path: str, visible: bool) -> None:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return
        token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        for prim in Usd.PrimRange(root):
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeVisible() if visible else imageable.MakeInvisible()
                # Explicitly set visibility token for determinism across nested assets.
                imageable.GetVisibilityAttr().Set(token)
                if visible:
                    try:
                        imageable.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
                    except Exception:
                        pass

    @staticmethod
    def _count_subtree_meshes(root_path: str) -> int:
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return 0
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return 0
        count = 0
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Mesh":
                count += 1
        return count

    @staticmethod
    def _set_root_uniform_scale(root_path: str, scale: float) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        if abs(scale - 1.0) < 1e-6:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        scale_vec = Gf.Vec3f(float(scale), float(scale), float(scale))
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(scale_vec)
                return
        xform.AddScaleOp().Set(scale_vec)

    @staticmethod
    def _hide_biped_template(biped) -> None:
        """Hide Biped_Setup template so it does not collide with the robot."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        path = "/World/Characters/Biped_Setup"
        if biped is not None and hasattr(biped, "GetPrimPath") and biped.IsValid():
            path = str(biped.GetPrimPath())
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        ops = list(xform.GetOrderedXformOps())
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0, 0, -100))
                break
        else:
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -100))
        img = UsdGeom.Imageable(prim)
        if img:
            img.MakeInvisible()
        parent_path = path.rsplit("/", 1)[0]
        if parent_path:
            parent = stage.GetPrimAtPath(parent_path)
            if parent.IsValid():
                img = UsdGeom.Imageable(parent)
                if img:
                    img.MakeInvisible()

    def _enforce_active_visibility(self) -> None:
        for slot_idx in self._active_slots:
            self._set_visibility(self._root_paths[slot_idx], True)
            if slot_idx < len(self._control_paths):
                self._set_visibility(self._control_paths[slot_idx], True)
            self._set_subtree_visibility(self._root_paths[slot_idx], True)

    def _set_character_pose(
        self,
        slot_idx: int,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self._root_prims[slot_idx].set_world_pose(position=position, orientation=orientation)
        if slot_idx < len(self._control_prims):
            self._control_prims[slot_idx].set_world_pose(position=position, orientation=orientation)

    @staticmethod
    def _set_visibility(path: str, visible: bool) -> None:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        img = UsdGeom.Imageable(prim)
        if img:
            if visible:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    @staticmethod
    def _find_driven_xform_path(root_path: str) -> str | None:
        """Find the prim under root that AnimGraph drives (ManRoot/asset with translate)."""
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        candidate: str | None = None
        for prim in Usd.PrimRange(root):
            if not prim.IsValid():
                continue
            path_str = str(prim.GetPrimPath())
            if "ManRoot" not in path_str:
                continue
            xform = UsdGeom.Xformable(prim)
            if not xform:
                continue
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    if candidate is None or len(path_str) > len(candidate):
                        candidate = path_str
                    break
        return candidate

    def _write_spawn_to_session(self, slot_idx: int, world_x: float, world_y: float, world_z: float) -> None:
        """Write spawn position to session layer so navigation sees correct start before GoTo."""
        import omni.usd
        from pxr import Gf, UsdGeom

        if slot_idx >= len(self._root_paths):
            return
        root_path = self._root_paths[slot_idx]
        driven_path = PeopleManager._find_driven_xform_path(root_path)
        if driven_path is None:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        session = stage.GetSessionLayer()
        if session is None or not session.identifier:
            return
        prim = stage.GetPrimAtPath(driven_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        parent = prim.GetParent()
        if not parent.IsValid():
            return
        parent_xform = UsdGeom.Xformable(parent)
        parent_world = parent_xform.ComputeLocalToWorldTransform(0)
        world = Gf.Matrix4d(1.0)
        world.SetTranslateOnly(Gf.Vec3d(world_x, world_y, world_z))
        local = parent_world.GetInverse() * world
        local_trans = local.ExtractTranslation()
        old_target = stage.GetEditTarget()
        old_id = old_target.GetLayer().identifier if old_target and old_target.GetLayer() else ""
        if not omni.usd.set_edit_target_by_identifier(stage, session.identifier):
            return
        try:
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    val = op.GetAttr().GetTypeName()
                    if "float" in str(val) or "Float" in str(val):
                        op.Set(Gf.Vec3f(local_trans[0], local_trans[1], local_trans[2]))
                    else:
                        op.Set(local_trans)
                    break
        finally:
            if old_id:
                omni.usd.set_edit_target_by_identifier(stage, old_id)

    def _enforce_z_override(self, env_origins: torch.Tensor) -> None:
        """Override Z on session layer so we win over AnimGraph; AnimGraph drives ManRoot/asset translate."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None or len(self._active_slots) == 0:
            return
        floor_z = float(getattr(self.cfg, "people_floor_z_world", 0.0))
        target_z = floor_z
        session = stage.GetSessionLayer()
        if session is None or not session.identifier:
            return
        old_target = stage.GetEditTarget()
        old_id = old_target.GetLayer().identifier if old_target and old_target.GetLayer() else ""
        if not omni.usd.set_edit_target_by_identifier(stage, session.identifier):
            return
        try:
            for slot_idx in self._active_slots:
                root_path = self._root_paths[slot_idx]
                driven_path = PeopleManager._find_driven_xform_path(root_path)
                if driven_path is None:
                    continue
                prim = stage.GetPrimAtPath(driven_path)
                if not prim.IsValid():
                    continue
                xform = UsdGeom.Xformable(prim)
                world = xform.ComputeLocalToWorldTransform(0)
                curr = world.ExtractTranslation()
                new_world = Gf.Matrix4d(world)
                new_world.SetTranslateOnly(Gf.Vec3d(curr[0], curr[1], target_z))
                parent = prim.GetParent()
                if not parent.IsValid():
                    continue
                parent_xform = UsdGeom.Xformable(parent)
                parent_world = parent_xform.ComputeLocalToWorldTransform(0)
                local = parent_world.GetInverse() * new_world
                local_trans = local.ExtractTranslation()
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        val = op.GetAttr().GetTypeName()
                        if "float" in str(val) or "Float" in str(val):
                            op.Set(Gf.Vec3f(local_trans[0], local_trans[1], local_trans[2]))
                        else:
                            op.Set(local_trans)
                        break
        finally:
            if old_id:
                omni.usd.set_edit_target_by_identifier(stage, old_id)

    def step(self, dt: float, env_origins: torch.Tensor) -> None:
        self._enforce_active_visibility()
        if not self._commands_enabled:
            return
        if len(self._active_slots) > 0:
            self._enforce_z_override(env_origins)
        self._command_refresh_accum_s += float(dt)
        refresh_s = float(getattr(self.cfg, "people_replan_interval_s", 2.0))
        if self._command_refresh_accum_s >= refresh_s:
            self._command_refresh_accum_s = 0.0
            self._inject_new_commands(env_origins)

    def _inject_new_commands(self, env_origins: torch.Tensor) -> None:
        if (self._agent_manager is None) or (not self._commands_enabled) or len(self._active_slots) == 0:
            return
        if not bool(getattr(self.cfg, "people_walk_enabled", True)):
            for flat_idx in self._active_slots:
                name = self._agent_names[flat_idx]
                try:
                    self._agent_manager.replace_command(name, [f"{name} Idle 999"])
                except Exception:
                    pass
            return
        floor_z = float(getattr(self.cfg, "people_floor_z_world", 0.0))
        target_z = floor_z
        nav_x_range, nav_y_range = getattr(self.cfg, "people_nav_xy_range", ((-7.0, 7.0), (-7.0, 7.0)))
        idle_min, idle_max = getattr(self.cfg, "people_idle_duration_range", (1.0, 3.0))
        cmd_count = int(getattr(self.cfg, "people_command_horizon", 3))
        for flat_idx in self._active_slots:
            env_id = self._env_id_for_slot[flat_idx] if flat_idx < len(self._env_id_for_slot) else 0
            origin = env_origins[env_id]
            name = self._agent_names[flat_idx]
            commands_for_agent: list[str] = []
            for k in range(cmd_count):
                gx = float(origin[0]) + float(torch.empty(1, device=self.device).uniform_(nav_x_range[0], nav_x_range[1]).item())
                gy = float(origin[1]) + float(torch.empty(1, device=self.device).uniform_(nav_y_range[0], nav_y_range[1]).item())
                commands_for_agent.append(f"{name} GoTo {gx:.3f} {gy:.3f} {target_z:.3f} _")
                if k < cmd_count - 1:
                    t = float(torch.empty(1, device=self.device).uniform_(idle_min, idle_max).item())
                    commands_for_agent.append(f"{name} Idle {t:.2f}")
            # In 5.1, repeated inject() can indefinitely grow command queue.
            # Replace queue each refresh to keep natural continuous locomotion.
            try:
                self._agent_manager.replace_command(name, commands_for_agent)
            except Exception:
                pass
        self._enforce_active_visibility()

    def reset(self, env_ids: Sequence[int] | torch.Tensor, env_origins: torch.Tensor) -> None:
        # Lazy IRA setup on first reset (covers terminal mode when sim starts on reset).
        if self._ira_setup_pending and not self._ira_setup_done:
            self.setup_after_sim_start()
        if len(self._root_paths) == 0:
            return

        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if len(env_ids_list) == 0:
            return

        floor_z = float(getattr(self.cfg, "people_floor_z_world", 0.0))
        spawn_z = floor_z

        min_count, max_count = getattr(self.cfg, "people_count_range", (0, 0))
        max_per_env = min(int(max_count), self._slots_per_env) if self._slots_per_env > 0 else min(int(max_count), len(self._root_paths))
        min_per_env = min(int(min_count), max_per_env)
        if max_per_env <= 0:
            return

        spawn_x_range, spawn_y_range = getattr(self.cfg, "people_spawn_xy_range", ((-6.0, 6.0), (-6.0, 6.0)))
        keepout = float(getattr(self.cfg, "people_keepout_radius", 1.5))
        min_sep = float(getattr(self.cfg, "people_min_separation", 1.2))
        max_tries = int(getattr(self.cfg, "people_max_sample_tries", 40))
        identity_quat = (1.0, 0.0, 0.0, 0.0)

        self._active_slots = set()
        self._command_refresh_accum_s = 0.0

        # Hide all and reset poses. For multi-env, use each env's origin for its characters.
        for flat_idx in range(len(self._root_paths)):
            env_id = self._env_id_for_slot[flat_idx] if flat_idx < len(self._env_id_for_slot) else 0
            origin = env_origins[env_id]
            hidden_pose = (float(origin[0]), float(origin[1]), -15.0)
            self._set_character_pose(flat_idx, hidden_pose, identity_quat)
            self._set_visibility(self._root_paths[flat_idx], False)
            if flat_idx < len(self._control_paths):
                self._set_visibility(self._control_paths[flat_idx], False)
            self._set_subtree_visibility(self._root_paths[flat_idx], False)

        # Per-env: randomly activate people_count_range people in each reset env.
        for env_id in env_ids_list:
            env_id = int(env_id)
            slot_start = env_id * self._slots_per_env
            slot_end = min(slot_start + self._slots_per_env, len(self._root_paths))
            env_slots = list(range(slot_start, slot_end))
            if len(env_slots) == 0:
                continue
            active_count = int(torch.randint(min_per_env, max_per_env + 1, (1,), device=self.device).item())
            active_count = min(active_count, len(env_slots))
            perm = torch.randperm(len(env_slots), device=self.device).tolist()
            requested_for_env = [env_slots[i] for i in perm[:active_count]]
            origin = env_origins[env_id]
            placed_xy: list[tuple[float, float]] = []

            for slot_idx in requested_for_env:
                sample_ok = False
                cand_x, cand_y = 0.0, 0.0
                for _ in range(max_tries):
                    cand_x = float(torch.empty(1, device=self.device).uniform_(spawn_x_range[0], spawn_x_range[1]).item())
                    cand_y = float(torch.empty(1, device=self.device).uniform_(spawn_y_range[0], spawn_y_range[1]).item())
                    if (cand_x * cand_x + cand_y * cand_y) < (keepout * keepout):
                        continue
                    too_close = any((cand_x - px) ** 2 + (cand_y - py) ** 2 < (min_sep * min_sep) for px, py in placed_xy)
                    if not too_close:
                        sample_ok = True
                        break
                if not sample_ok:
                    continue

                spawn_x = float(origin[0]) + cand_x
                spawn_y = float(origin[1]) + cand_y

                yaw = float(torch.empty(1, device=self.device).uniform_(-math.pi, math.pi).item())
                half = 0.5 * yaw
                yaw_quat = (math.cos(half), 0.0, 0.0, math.sin(half))
                bq = self._base_quat_wxyz[slot_idx]
                w = yaw_quat[0] * bq[0] - yaw_quat[1] * bq[1] - yaw_quat[2] * bq[2] - yaw_quat[3] * bq[3]
                x = yaw_quat[0] * bq[1] + yaw_quat[1] * bq[0] + yaw_quat[2] * bq[3] - yaw_quat[3] * bq[2]
                y = yaw_quat[0] * bq[2] - yaw_quat[1] * bq[3] + yaw_quat[2] * bq[0] + yaw_quat[3] * bq[1]
                z = yaw_quat[0] * bq[3] + yaw_quat[1] * bq[2] - yaw_quat[2] * bq[1] + yaw_quat[3] * bq[0]
                quat_wxyz = (w, x, y, z)
                world_pos = (spawn_x, spawn_y, spawn_z)
                self._set_character_pose(slot_idx, world_pos, quat_wxyz)
                self._write_spawn_to_session(slot_idx, spawn_x, spawn_y, spawn_z)
                self._set_visibility(self._root_paths[slot_idx], True)
                if slot_idx < len(self._control_paths):
                    self._set_visibility(self._control_paths[slot_idx], True)
                self._set_subtree_visibility(self._root_paths[slot_idx], True)
                placed_xy.append((cand_x, cand_y))
                self._active_slots.add(slot_idx)
        # Inject short command chains so visible characters keep moving.
        if (self._agent_manager is None) or (not self._commands_enabled):
            return
        if self._navmesh_ready:
            self._spin_app_updates(8)
            self._inject_new_commands(env_origins)
