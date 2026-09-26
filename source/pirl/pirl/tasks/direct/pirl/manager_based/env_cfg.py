from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from ..pirl_env_cfg import PirlTaskCfg
from ..pirl_env_dyn_obstacles import build_collection_cfg
from .mdp import events, observations, rewards, terminations
from .mdp.actions import DifferentialDriveAction, DifferentialDriveActionCfg

# ``@configclass`` turns annotations into dataclass fields, so task constants
# such as ``robot_cfg`` are instance attributes rather than class attributes.
_TASK_CFG = PirlTaskCfg()


@configclass
class PirlSceneCfg(InteractiveSceneCfg):
    # This plane is global (not cloned under each ``env_*`` namespace).  The
    # default is only 100 m square, whereas a 50-env grid with 35 m spacing
    # spans about 245 m; robots outside it fall through the world.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(300.0, 300.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(_TASK_CFG.ground_static_friction),
                dynamic_friction=float(_TASK_CFG.ground_dynamic_friction),
                friction_combine_mode="max",
            ),
        ),
    )
    robot = _TASK_CFG.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]
    dyn_obstacles = build_collection_cfg(_TASK_CFG)
    lidar = _TASK_CFG.lidar.replace(  # type: ignore[attr-defined]
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/GroundPlane"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/DynObstacle_.*", track_mesh_transforms=True
            ),
        ]
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        vec = ObservationTermCfg(func=observations.vector)
        costmap = ObservationTermCfg(func=observations.costmap)

        def __post_init__(self) -> None:
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    drive: DifferentialDriveActionCfg = DifferentialDriveActionCfg(
        asset_name="robot", class_type=DifferentialDriveAction
    )


@configclass
class RewardsCfg:
    progress = RewardTermCfg(func=rewards.progress, weight=float(_TASK_CFG.rew_scale_progress))
    path_error = RewardTermCfg(func=rewards.path_error, weight=float(_TASK_CFG.rew_scale_path_error))
    heading = RewardTermCfg(func=rewards.heading, weight=float(_TASK_CFG.rew_scale_heading))
    collision = RewardTermCfg(func=rewards.collision, weight=float(_TASK_CFG.rew_scale_collision))
    time = RewardTermCfg(func=rewards.time, weight=float(_TASK_CFG.rew_scale_time))
    success = RewardTermCfg(func=rewards.success, weight=float(_TASK_CFG.rew_scale_success))


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)
    collision = TerminationTermCfg(func=terminations.collision)
    success = TerminationTermCfg(func=terminations.success)


@configclass
class EventsCfg:
    reset_navigation = EventTermCfg(func=events.reset_navigation, mode="reset")


@configclass
class PirlManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene: PirlSceneCfg = PirlSceneCfg(num_envs=50, env_spacing=35.0, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        self.decimation = _TASK_CFG.decimation
        self.episode_length_s = _TASK_CFG.episode_length_s
        self.sim.dt = _TASK_CFG.physics_dt
        self.sim.render_interval = self.decimation
