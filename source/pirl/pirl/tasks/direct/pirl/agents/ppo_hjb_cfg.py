"""PPO-RNN + HJB agent configuration (skrl 2.x dataclass)."""

from __future__ import annotations

import dataclasses

from skrl.agents.torch.ppo import PPO_CFG


@dataclasses.dataclass(kw_only=True)
class PPOHjbRNN_CFG(PPO_CFG):
    """PPO-RNN configuration extended with HJB critic regularizer fields."""

    hjb_loss_scale: float = 0.5
    hjb_max_lin_vel: float = 0.5
    hjb_max_ang_vel: float = 1.5
    hjb_time_weight: float = 0.5
    hjb_distance_weight: float = 0.2
    hjb_heading_weight: float = 0.2
    hjb_control_weight: float = 0.05
    hjb_progress_weight: float = 1.0
    hjb_hamiltonian_mode: str = "optimal"
    hjb_vec_d_index: int = 2
    hjb_vec_psi_index: int = 3
    hjb_step_dt: float = 1.0 / 60.0
    hjb_lidar_hits_start_index: int = -1
    hjb_lidar_sector_count: int = 0
